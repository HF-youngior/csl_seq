import torch
import torch.nn as nn
import torchvision.models as models
import random

import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)

"""
Implementation of Sequence to Sequence Model with (2+1)D ResNet
Encoder: (2+1)D ResNet + LSTM
Decoder: LSTM
"""
class ResNet2plus1D(nn.Module):
    """
    (2+1)D ResNet编码器
    结合2D空间卷积和1D时序卷积
    """
    def __init__(self, arch="resnet18", pretrained=True):
        super(ResNet2plus1D, self).__init__()
        
        # 加载预训练的2D ResNet
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        
        # 移除最后的全连接层
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # 添加1D时序卷积层（增加特征维度）
        self.temporal_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim * 2,  # 增加特征维度
            kernel_size=3,
            padding=1
        )
        
        # 批归一化
        self.bn = nn.BatchNorm1d(self.feature_dim * 2)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # 更新特征维度
        self.feature_dim *= 2
        
    def forward(self, x):
        """
        x: [batch_size, 3, seq_len, h, w]
        返回: [batch_size, seq_len, feature_dim]
        """
        batch_size, channels, seq_len, height, width = x.size()
        
        # 提取每一帧的特征
        features = []
        for t in range(seq_len):
            frame = x[:, :, t, :, :]  # [batch_size, 3, h, w]
            feat = self.resnet(frame)  # [batch_size, feature_dim/2, 1, 1]
            feat = feat.view(batch_size, -1)  # [batch_size, feature_dim/2]
            features.append(feat)
        
        # 堆叠特征序列
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, feature_dim/2]
        
        # 转换为1D卷积的输入格式
        features = features.transpose(1, 2)  # [batch_size, feature_dim/2, seq_len]
        
        # 1D时序卷积
        temporal_features = self.temporal_conv(features)
        temporal_features = self.bn(temporal_features)
        temporal_features = self.relu(temporal_features)
        
        # 转换回序列格式
        temporal_features = temporal_features.transpose(1, 2)  # [batch_size, seq_len, feature_dim]
        
        return temporal_features

class Encoder(nn.Module):
    """
    编码器: (2+1)D ResNet + LSTM
    """
    def __init__(self, lstm_hidden_size=512, arch="resnet18"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        
        # (2+1)D ResNet特征提取器
        self.resnet_2plus1d = ResNet2plus1D(arch=arch, pretrained=True)
        
        # LSTM层（输入维度是时序卷积后的特征维度）
        self.lstm = nn.LSTM(
            input_size=self.resnet_2plus1d.feature_dim,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )

    def forward(self, x):
        """
        x: [batch_size, channels, seq_len, h, w]
        """
        # (2+1)D ResNet特征提取
        temporal_features = self.resnet_2plus1d(x)  # [batch_size, seq_len, feature_dim]
        
        # LSTM处理
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(temporal_features, None)
        
        # 返回所有隐藏状态和最终状态
        return out, (h_n.squeeze(0), c_n.squeeze(0))

class Decoder(nn.Module):
    """
    解码器: LSTM
    """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim+enc_hid_dim, dec_hid_dim)
        self.fc = nn.Linear(emb_dim+enc_hid_dim+dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, context):
        """
        input: [batch_size] - 上一个预测的词
        hidden: [batch_size, dec_hid_dim] - 解码器隐藏状态
        cell: [batch_size, dec_hid_dim] - 解码器细胞状态
        context: [batch_size, enc_hid_dim] - 上下文向量
        """
        # 扩展维度
        input = input.unsqueeze(0)

        # 嵌入
        embedded = self.dropout(self.embedding(input))

        # 拼接嵌入和上下文
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)

        # LSTM
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # 压缩维度
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)

        # 预测
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))

        return prediction, (hidden, cell)

class Seq2Seq_2plus1D(nn.Module):
    """
    Seq2Seq模型: (2+1)D ResNet编码器 + LSTM解码器
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq_2plus1D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        """
        imgs: [batch_size, channels, seq_len, h, w]
        target: [batch_size, trg_len]
        """
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # 存储解码器输出
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 编码器输出: [batch, seq_len, hidden_size]
        encoder_outputs, (hidden, cell) = self.encoder(imgs)

        # 计算上下文向量
        context = encoder_outputs.mean(dim=1)

        # 解码器的第一个输入是<sos> token
        input = target[:, 0]

        for t in range(1, trg_len):
            # 解码
            output, (hidden, cell) = self.decoder(input, hidden, cell, context)

            # 存储预测
            outputs[t] = output

            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio

            # 获取最高预测token
            top1 = output.argmax(1)

            # 应用教师强制
            input = target[:, t] if teacher_force else top1

        return outputs

# Test
if __name__ == '__main__':
    # 测试编码器
    encoder = Encoder(lstm_hidden_size=512, arch="resnet18")
    imgs = torch.randn(16, 3, 8, 128, 128)
    print("Testing Encoder...")
    encoder_outputs, (hidden, cell) = encoder(imgs)
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Hidden shape: {hidden.shape}, Cell shape: {cell.shape}")

    # 测试解码器
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    input = torch.LongTensor(16).random_(0, 500)
    hidden = torch.randn(16, 512)
    cell = torch.randn(16, 512)
    context = torch.randn(16, 512)
    print("\nTesting Decoder...")
    output, (hidden, cell) = decoder(input, hidden, cell, context)
    print(f"Decoder output shape: {output.shape}")

    # 测试完整模型
    device = torch.device("cpu")
    seq2seq = Seq2Seq_2plus1D(encoder=encoder, decoder=decoder, device=device)
    imgs = torch.randn(16, 3, 8, 128, 128)
    target = torch.LongTensor(16, 8).random_(0, 500)
    print("\nTesting Full Model...")
    outputs = seq2seq(imgs, target)
    print(f"Model outputs shape: {outputs.shape}")
    predictions = outputs.argmax(dim=2).permute(1, 0)  # batch first
    print(f"Predictions shape: {predictions.shape}")
