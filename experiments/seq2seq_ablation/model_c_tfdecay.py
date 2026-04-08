from models.Seq2Seq import Decoder, Encoder, Seq2Seq


def compute_teacher_forcing_ratio(epoch_idx, tf_start, tf_end, tf_decay_epochs):
    decay_epochs = int(max(1, tf_decay_epochs))
    if epoch_idx >= decay_epochs:
        return float(tf_end)
    progress = float(epoch_idx) / float(max(1, decay_epochs - 1))
    return float(tf_start) + (float(tf_end) - float(tf_start)) * progress


def build_model(output_dim, enc_hid_dim, dec_hid_dim, emb_dim, dropout, device):
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=emb_dim,
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        dropout=dropout,
    ).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    return model

