import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        # 直接使用列表计算，避免PyTorch张量操作的问题
        feat_len_list = []
        for length in lgt.cpu().numpy():
            current_length = length
            for ks in self.kernel_size:
                if ks[0] == 'P':
                    current_length = current_length // 2
                else:
                    current_length -= int(ks[1]) - 1
            # 确保长度始终大于0
            current_length = max(1, current_length)
            feat_len_list.append(current_length)
        # 转换回张量
        return torch.tensor(feat_len_list, device=lgt.device, dtype=torch.int64)

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        if self.num_classes == -1:
            logits = None
        else:
            logits = self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt,
        }
