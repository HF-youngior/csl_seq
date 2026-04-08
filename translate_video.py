#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语视频翻译脚本
使用训练好的模型将手语视频翻译成中文文字
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.Seq2Seq import Encoder, Decoder, Seq2Seq
from dataset import CSL_Continuous, CSL_Continuous_Char

class SignLanguageTranslator:
    """手语翻译器"""
    
    def __init__(self, model_path, dict_path, corpus_path, use_char_level=False, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_char_level = use_char_level
        
        # 加载字典和语料库
        self.load_vocabulary(dict_path, corpus_path)
        
        # 创建模型
        self.create_model()
        
        # 加载训练好的权重
        self.load_model(model_path)
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def load_vocabulary(self, dict_path, corpus_path):
        """加载词汇表"""
        # 这里需要根据您的字典和语料库格式来加载
        # 简化版本，实际使用时需要完整实现
        self.word_to_id = {}
        self.id_to_word = {}
        
        # 特殊标记
        self.word_to_id['<pad>'] = 0
        self.word_to_id['<sos>'] = 1
        self.word_to_id['<eos>'] = 2
        self.id_to_word[0] = '<pad>'
        self.id_to_word[1] = '<sos>'
        self.id_to_word[2] = '<eos>'
        
        # 加载字典
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word_id = int(parts[0])
                    word = parts[1]
                    self.word_to_id[word] = word_id + 3  # 偏移3个特殊标记
                    self.id_to_word[word_id + 3] = word
    
    def create_model(self):
        """创建模型"""
        # 创建编码器和解码器
        self.encoder = Encoder(lstm_hidden_size=512, arch="resnet18").to(self.device)
        self.decoder = Decoder(
            output_dim=len(self.word_to_id),
            emb_dim=256,
            enc_hid_dim=512,
            dec_hid_dim=512,
            dropout=0.5
        ).to(self.device)
        
        # 创建Seq2Seq模型
        self.model = Seq2Seq(
            encoder=self.encoder,
            decoder=self.decoder,
            device=self.device
        ).to(self.device)
    
    def load_model(self, model_path):
        """加载训练好的模型权重"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型已加载: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model.eval()
    
    def video_to_frames(self, video_path, num_frames=48):
        """将视频转换为帧序列"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样帧
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # 转换BGR到RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为PIL Image
                frame = Image.fromarray(frame)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def frames_to_tensor(self, frames):
        """将帧序列转换为张量"""
        processed_frames = []
        for frame in frames:
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
        
        # 堆叠成张量 (C, T, H, W)
        frames_tensor = torch.stack(processed_frames, dim=1)
        # 添加batch维度 (1, C, T, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor.to(self.device)
    
    def translate_video(self, video_path):
        """翻译手语视频"""
        print(f"正在处理视频: {video_path}")
        
        # 1. 提取视频帧
        frames = self.video_to_frames(video_path)
        if len(frames) == 0:
            return "错误：无法从视频中提取帧"
        
        # 2. 转换为张量
        video_tensor = self.frames_to_tensor(frames)
        
        # 3. 模型推理
        with torch.no_grad():
            # 创建虚拟目标序列（用于模型前向传播）
            batch_size = video_tensor.size(0)
            max_length = 50  # 假设最大输出长度为50
            dummy_target = torch.zeros(batch_size, max_length, dtype=torch.long).to(self.device)
            
            # 模型前向传播
            outputs = self.model(video_tensor, dummy_target, 0)  # 0表示不使用teacher forcing
            
            # 获取预测结果
            predictions = torch.argmax(outputs, dim=-1)  # (seq_len, batch_size)
            predictions = predictions.squeeze(1)  # (seq_len,)
        
        # 4. 转换为文字
        words = []
        for pred_id in predictions:
            if pred_id.item() in self.id_to_word:
                word = self.id_to_word[pred_id.item()]
                if word not in ['<pad>', '<sos>', '<eos>']:
                    words.append(word)
                elif word == '<eos>':
                    break  # 遇到结束标记就停止
        
        # 5. 组合成句子
        translation = ''.join(words) if self.use_char_level else ' '.join(words)
        
        return translation

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='手语视频翻译')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--dict_path', type=str, default='dictionary.txt', help='字典文件路径')
    parser.add_argument('--corpus_path', type=str, default='corpus.txt', help='语料库文件路径')
    parser.add_argument('--use_char_level', action='store_true', help='使用字符级别翻译')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误：视频文件不存在: {args.video_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件不存在: {args.model_path}")
        return
    
    # 创建翻译器
    try:
        translator = SignLanguageTranslator(
            model_path=args.model_path,
            dict_path=args.dict_path,
            corpus_path=args.corpus_path,
            use_char_level=args.use_char_level,
            device=args.device
        )
        
        # 翻译视频
        translation = translator.translate_video(args.video_path)
        
        print(f"\n翻译结果: {translation}")
        
    except Exception as e:
        print(f"翻译过程中出现错误: {e}")

if __name__ == '__main__':
    main()

