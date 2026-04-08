import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random

import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)

"""
Simplified Sequence to Sequence Model with Attention
Encoder: ResNet + LSTM with attention
Decoder: LSTM with attention mechanism
"""
class Attention(nn.Module):
    """
    注意力机制模块
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, dec_hid_dim]
        encoder_outputs: [batch_size, src_len, enc_hid_dim]
        返回: [batch_size, src_len] 注意力权重
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 扩展hidden维度以匹配encoder_outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # 计算注意力能量
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # 计算注意力权重
        attention_weights = F.softmax(attention, dim=1)
        
        return attention_weights

class Encoder(nn.Module):
    """
    编码器：ResNet + LSTM + LayerNorm
    """
    def __init__(self, lstm_hidden_size=512, arch="resnet18"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        
        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.lstm_hidden_size)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            out = self.resnet(x[:, :, t, :, :])
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        
        # 层归一化
        out = self.layer_norm(out)
        
        # num_layers * num_directions = 1
        return out, (h_n.squeeze(0), c_n.squeeze(0))

class Decoder(nn.Module):
    """
    解码器：LSTM + 注意力机制 + LayerNorm
    """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # 注意力机制
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        
        # LSTM层
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim)
        
        # 输出层
        self.fc = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(dec_hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        input: [batch_size] - 上一个预测的词
        hidden: [batch_size, dec_hid_dim] - 解码器隐藏状态
        cell: [batch_size, dec_hid_dim] - 解码器细胞状态
        encoder_outputs: [batch_size, src_len, enc_hid_dim] - 编码器输出
        """
        # expand dim to (1, batch_size)
        input = input.unsqueeze(0)

        # embedded(1, batch_size, emb_dim): embed last prediction word
        embedded = self.dropout(self.embedding(input))

        # 计算注意力权重
        attention_weights = self.attention(hidden, encoder_outputs)
        
        # 计算上下文向量（加权和）
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # rnn_input(1, batch_size, emb_dim+enc_hid_dim): concat embedded and context 
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)

        # LSTM
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # hidden(batch_size, dec_hid_dim)
        # cell(batch_size, dec_hid_dim)
        # embedded(1, batch_size, emb_dim)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)
        
        # 层归一化
        hidden = self.layer_norm(hidden)

        # prediction
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))

        return prediction, (hidden, cell)

class Seq2Seq_v3(nn.Module):
    """
    简化版Seq2Seq模型：带注意力机制的编码器-解码器
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq_v3, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        # imgs: (batch_size, channels, T, H, W)
        # target: (batch_size, trg len)
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs(batch, seq_len, hidden_size): all hidden states of input sequence
        encoder_outputs, (hidden, cell) = self.encoder(imgs)

        # first input to the decoder is the <sos> tokens
        input = target[:, 0]

        for t in range(1, trg_len):
            # decode with attention
            output, (hidden, cell) = self.decoder(input, hidden, cell, encoder_outputs)

            # store prediction
            outputs[t] = output

            # decide whether to do teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            input = target[:, t] if teacher_force else top1

        return outputs

# Test
if __name__ == '__main__':
    # test encoder
    encoder = Encoder(lstm_hidden_size=512)
    imgs = torch.randn(16, 3, 8, 128, 128)
    print("Testing Encoder...")
    encoder_outputs, (hidden, cell) = encoder(imgs)
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Hidden shape: {hidden.shape}, Cell shape: {cell.shape}")

    # test decoder
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    input = torch.LongTensor(16).random_(0, 500)
    hidden = torch.randn(16, 512)
    cell = torch.randn(16, 512)
    encoder_outputs = torch.randn(16, 8, 512)
    print("\nTesting Decoder...")
    output, (hidden, cell) = decoder(input, hidden, cell, encoder_outputs)
    print(f"Decoder output shape: {output.shape}")

    # test seq2seq
    device = torch.device("cpu")
    seq2seq = Seq2Seq_v3(encoder=encoder, decoder=decoder, device=device)
    imgs = torch.randn(16, 3, 8, 128, 128)
    target = torch.LongTensor(16, 8).random_(0, 500)
    print("\nTesting Full Model...")
    outputs = seq2seq(imgs, target)
    print(f"Model outputs shape: {outputs.shape}")
    predictions = outputs.argmax(dim=2).permute(1, 0)  # batch first
    print(f"Predictions shape: {predictions.shape}")
