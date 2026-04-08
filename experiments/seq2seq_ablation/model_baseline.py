from models.Seq2Seq import Decoder, Encoder, Seq2Seq


def build_model(output_dim, enc_hid_dim, dec_hid_dim, emb_dim, dropout, device):
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=emb_dim,
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        dropout=dropout,
    ).to(device)
    return encoder, decoder, Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
