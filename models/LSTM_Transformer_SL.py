import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    位置编码层，为输入序列添加位置信息
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VideoFeatureExtractor(nn.Module):
    """
    视频特征提取器，使用3D CNN或2D CNN+时序编码
    """
    def __init__(self, d_model, pretrained=True):
        super(VideoFeatureExtractor, self).__init__()
        # 使用ResNet18作为基础网络
        from torchvision.models import resnet18
        self.resnet = resnet18(pretrained=pretrained)
        # 移除最后的分类层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # 调整特征维度
        self.fc = nn.Linear(512, d_model)
    
    def forward(self, x):
        # x: [batch_size, 3, seq_len, 128, 128]
        batch_size, channels, seq_len, height, width = x.size()
        features = []
        
        # 对每一帧提取特征
        for t in range(seq_len):
            frame = x[:, :, t, :, :]  # [batch_size, 3, 128, 128]
            feat = self.resnet(frame)  # [batch_size, 512, 1, 1]
            feat = feat.view(batch_size, -1)  # [batch_size, 512]
            feat = self.fc(feat)  # [batch_size, d_model]
            features.append(feat)
        
        # 堆叠特征序列
        features = torch.stack(features, dim=0)  # [seq_len, batch_size, d_model]
        return features

class LSTMEncoder(nn.Module):
    """
    LSTM编码器，用于提取视频的时序特征
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=False
        )
        
        # 调整双向LSTM的输出维度
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
    
    def forward(self, src_features):
        """
        src_features: [seq_len, batch_size, input_dim]
        """
        # LSTM前向传播
        outputs, (hidden, cell) = self.lstm(src_features)
        
        # 处理隐藏状态和细胞状态（双向LSTM需要特殊处理）
        if self.bidirectional:
            # 将双向LSTM的隐藏状态和细胞状态分别拼接
            hidden = torch.cat((hidden[::2], hidden[1::2]), dim=2)  # [num_layers, batch_size, hidden_dim*2]
            cell = torch.cat((cell[::2], cell[1::2]), dim=2)  # [num_layers, batch_size, hidden_dim*2]
        
        return outputs, hidden, cell

class LSTMTransformerSL(nn.Module):
    """
    连续手语识别的混合模型：LSTM编码器 + Transformer解码器
    """
    def __init__(self, 
                 num_classes, 
                 d_model=512, 
                 hidden_dim=256, 
                 num_layers=2,
                 nhead=8, 
                 num_decoder_layers=4, 
                 dim_feedforward=1024, 
                 dropout=0.3):
        super(LSTMTransformerSL, self).__init__()
        
        # 视频特征提取器
        self.feature_extractor = VideoFeatureExtractor(d_model)
        
        # LSTM编码器
        self.encoder = LSTMEncoder(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # 编码器输出维度到Transformer维度的映射
        self.encoder_output_proj = nn.Linear(self.encoder.output_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 目标序列嵌入
        self.tgt_embedding = nn.Embedding(num_classes, d_model)
        
        # 时序信息处理层（添加到Transformer解码器中）
        self.temporal_processor = nn.LSTM(
            d_model,
            d_model,
            num_layers=1,
            batch_first=False,
            dropout=dropout
        )
        
        # Transformer解码器
        decoder_layers = nn.TransformerDecoderLayer(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # 输出层
        self.fc_out = nn.Linear(d_model, num_classes)
        
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        前向传播
        src: [batch_size, 3, seq_len, 128, 128] - 视频输入
        tgt: [batch_size, tgt_len] - 目标序列
        """
        # 提取视频特征
        src_features = self.feature_extractor(src)  # [seq_len, batch_size, d_model]
        
        # LSTM编码
        encoder_outputs, hidden, cell = self.encoder(src_features)
        
        # 调整编码器输出维度
        encoder_outputs = self.encoder_output_proj(encoder_outputs)  # [seq_len, batch_size, d_model]
        
        # 添加位置编码
        encoder_outputs = self.pos_encoder(encoder_outputs)
        
        # 目标序列嵌入
        tgt_embedded = self.tgt_embedding(tgt).transpose(0, 1)  # [tgt_len, batch_size, d_model]
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # 生成解码器掩码
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(src.device)
        
        # 添加时序信息处理（在解码器之前）
        tgt_embedded, _ = self.temporal_processor(tgt_embedded)
        
        # Transformer解码
        output = self.transformer_decoder(
            tgt_embedded, 
            encoder_outputs, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # 输出层
        output = self.fc_out(output)  # [tgt_len, batch_size, num_classes]
        
        return output.transpose(0, 1)  # [batch_size, tgt_len, num_classes]
    
    def generate_square_subsequent_mask(self, sz):
        """
        生成正方形的后续掩码，用于解码器的自注意力层
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def predict(self, src, max_len=50, start_token=1, end_token=2):
        """
        推理模式，用于生成预测结果
        """
        self.eval()
        
        batch_size = src.size(0)
        device = src.device
        
        # 提取视频特征
        src_features = self.feature_extractor(src)
        
        # LSTM编码
        encoder_outputs, hidden, cell = self.encoder(src_features)
        encoder_outputs = self.encoder_output_proj(encoder_outputs)
        encoder_outputs = self.pos_encoder(encoder_outputs)
        
        # 初始化目标序列
        output = torch.zeros(batch_size, max_len, dtype=torch.long).to(device)
        output[:, 0] = start_token
        
        # 自回归生成
        for i in range(1, max_len):
            tgt = output[:, :i]
            tgt_embedded = self.tgt_embedding(tgt).transpose(0, 1)
            tgt_embedded = self.pos_encoder(tgt_embedded)
            
            # 添加时序信息处理
            tgt_embedded, _ = self.temporal_processor(tgt_embedded)
            
            tgt_mask = self.generate_square_subsequent_mask(i).to(device)
            
            decoder_output = self.transformer_decoder(
                tgt_embedded, 
                encoder_outputs,
                tgt_mask=tgt_mask
            )
            
            prob = self.fc_out(decoder_output[-1, :, :])
            _, next_word = torch.max(prob, dim=1)
            
            output[:, i] = next_word
            
            # 如果所有序列都已生成结束符，提前终止
            if (next_word == end_token).all():
                break
        
        return output
