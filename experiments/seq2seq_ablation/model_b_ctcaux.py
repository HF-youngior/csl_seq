import random

import torch
import torch.nn as nn

from models.Seq2Seq import Decoder, Encoder


class Seq2SeqWithCTCAux(nn.Module):
    def __init__(self, encoder, decoder, output_dim, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.ctc_head = nn.Linear(encoder.lstm_hidden_size, output_dim)

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(imgs)
        context = encoder_outputs.mean(dim=1)
        input_token = target[:, 0]

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input_token, hidden, cell, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = target[:, t] if teacher_force else top1

        ctc_logits = self.ctc_head(encoder_outputs)
        return outputs, ctc_logits


def build_model(output_dim, enc_hid_dim, dec_hid_dim, emb_dim, dropout, device):
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=emb_dim,
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        dropout=dropout,
    ).to(device)
    model = Seq2SeqWithCTCAux(
        encoder=encoder,
        decoder=decoder,
        output_dim=output_dim,
        device=device,
    ).to(device)
    return model

