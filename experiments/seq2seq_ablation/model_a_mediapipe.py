import random

import torch
import torch.nn as nn

from models.Seq2Seq import Decoder, Encoder


class Seq2SeqPoseFusion(nn.Module):
    def __init__(self, encoder, decoder, device, pose_dim, pose_fuse_weight=0.2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pose_dim = int(max(0, pose_dim))
        self.pose_fuse_weight = float(pose_fuse_weight)
        if self.pose_dim > 0:
            self.pose_proj = nn.Linear(self.pose_dim, encoder.lstm_hidden_size)
            self.pose_gate = nn.Linear(encoder.lstm_hidden_size, encoder.lstm_hidden_size)
        else:
            self.pose_proj = None
            self.pose_gate = None

    def _fuse_context(self, encoder_outputs, pose_feat):
        context = encoder_outputs.mean(dim=1)
        if self.pose_proj is None:
            return context
        if pose_feat is None or pose_feat.numel() == 0:
            return context
        pose_emb = torch.tanh(self.pose_proj(pose_feat))
        gate = torch.sigmoid(self.pose_gate(pose_emb))
        return context + self.pose_fuse_weight * gate * pose_emb

    def forward(self, imgs, target, pose_feat=None, teacher_forcing_ratio=0.5):
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(imgs)
        context = self._fuse_context(encoder_outputs, pose_feat)
        input_token = target[:, 0]

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input_token, hidden, cell, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = target[:, t] if teacher_force else top1
        return outputs


def build_model(output_dim, enc_hid_dim, dec_hid_dim, emb_dim, dropout, device, pose_dim, pose_fuse_weight):
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=emb_dim,
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        dropout=dropout,
    ).to(device)
    model = Seq2SeqPoseFusion(
        encoder=encoder,
        decoder=decoder,
        device=device,
        pose_dim=pose_dim,
        pose_fuse_weight=pose_fuse_weight,
    ).to(device)
    return model

