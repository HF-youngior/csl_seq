import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, ctc_zero_infinity=True,
            ctc_disable_cudnn=True, ctc_on_cpu=False, backward_nan_filter=False,
            decode_mode='max', decode_beam_width=10, decode_num_processes=10
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init(ctc_zero_infinity=ctc_zero_infinity)
        self.ctc_disable_cudnn = bool(ctc_disable_cudnn)
        self.ctc_on_cpu = bool(ctc_on_cpu)
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(
            gloss_dict,
            num_classes,
            decode_mode,
            beam_width=decode_beam_width,
            num_processes=decode_num_processes,
        )
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        if backward_nan_filter:
            self.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            if g is not None:
                g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        # inputs: [batch*temp, 3, 224, 224]
        # 通过conv2d提取所有帧的特征
        features = self.conv2d(inputs)
        return features

    def forward(self, x, len_x, label=None, label_lgt=None, return_conv_decode=True):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None
        if (not self.training) and return_conv_decode:
            conv_pred = self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        input_lengths = ret_dict["feat_len"].cpu().int()
        target_lengths = label_lgt.cpu().int()
        targets = label.cpu().int()
        conv_logits = torch.nan_to_num(ret_dict["conv_logits"], nan=0.0, posinf=1e4, neginf=-1e4)
        seq_logits = torch.nan_to_num(ret_dict["sequence_logits"], nan=0.0, posinf=1e4, neginf=-1e4)

        def _ctc_loss(logits):
            log_probs = logits.log_softmax(-1)
            if self.ctc_on_cpu:
                log_probs_cpu = log_probs.float().cpu()
                targets_cpu = targets.cpu()
                input_lengths_cpu = input_lengths.cpu()
                target_lengths_cpu = target_lengths.cpu()
                ctc_cpu = self.loss['CTCLoss'](
                    log_probs_cpu,
                    targets_cpu,
                    input_lengths_cpu,
                    target_lengths_cpu,
                ).mean()
                return ctc_cpu.to(log_probs.device)
            with torch.backends.cudnn.flags(enabled=not self.ctc_disable_cudnn):
                return self.loss['CTCLoss'](
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                ).mean()

        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * _ctc_loss(conv_logits)
            elif k == 'SeqCTC':
                loss += weight * _ctc_loss(seq_logits)
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](conv_logits,
                                                           seq_logits.detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self, ctc_zero_infinity=True):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(
            reduction='none',
            zero_infinity=bool(ctc_zero_infinity),
        )
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
