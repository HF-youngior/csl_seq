#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续手语识别主程序 - 优化版Seq2Seq_v4
基于原始Seq2Seq模型的最小化优化版本
"""

import os
import sys
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import argparse
import editdistance

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import CSL_Continuous, CSL_Continuous_Char
from models.Seq2Seq_v4 import Encoder, Decoder, Seq2Seq_v4
from sklearn.metrics import accuracy_score

def get_data_transform(sample_size, is_training=True):
    """
    获取数据变换（与原始模型保持一致）
    """
    return transforms.Compose([
        transforms.Resize([sample_size, sample_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def calculate_wer(predicted, target):
    """
    计算词错误率 (Word Error Rate)
    移除特殊标记（0,1,2 - padding, sos, eos）
    """
    # 移除特殊标记
    pred_clean = [item for item in predicted if item not in [0, 1, 2]]
    target_clean = [item for item in target if item not in [0, 1, 2]]
    
    if len(target_clean) == 0:
        return 0
    
    # 计算编辑距离
    return editdistance.eval(pred_clean, target_clean) / len(target_clean)

def train_model(args, model, train_loader, val_loader, criterion, optimizer, device, writer, log_file, train_dataset):
    """
    训练模型（与原始训练脚本保持一致）
    """
    best_val_acc = 0.0
    best_model_path = os.path.join(args.model_path, 'best_model.pth')
    
    for epoch in range(args.epochs):
        model.train()
        losses = []
        all_trg = []
        all_pred = []
        all_wer = []
        
        print(f'\nepoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (imgs, target) in enumerate(train_loader):
            imgs = imgs.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # 模型前向传播
            outputs = model(imgs, target, teacher_forcing_ratio=args.teacher_forcing_ratio)
            
            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[1:].view(-1, output_dim)
            target_flat = target.permute(1, 0)[1:].reshape(-1)
            
            # 计算损失
            loss = criterion(outputs_flat, target_flat)
            losses.append(loss.item())
            
            # 计算准确率（与原始脚本相同）
            prediction = torch.max(outputs_flat, 1)[1]
            score = accuracy_score(target_flat.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
            all_trg.extend(target_flat)
            all_pred.extend(prediction)
            
            # 计算WER（与原始脚本相同）
            batch_size = imgs.shape[0]
            prediction_seq = prediction.view(-1, batch_size).permute(1, 0).tolist()
            target_seq = target_flat.view(-1, batch_size).permute(1, 0).tolist()
            wers = []
            for i in range(batch_size):
                # add mask(remove padding, sos, eos)
                wers.append(calculate_wer(prediction_seq[i], target_seq[i]))
            all_wer.extend(wers)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = loss.item()
                avg_wer = sum(wers) / len(wers) if len(wers) > 0 else 0
                
                print(f"epoch {epoch+1} | iteration {batch_idx+1} | Loss {avg_loss:.6f} | Acc {score*100:.2f}% | WER {avg_wer:.2f}%")
                
                # 写入TensorBoard
                writer.add_scalar('train/loss', avg_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/accuracy', score*100, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/wer', avg_wer, epoch * len(train_loader) + batch_idx)
        
        # 计算平均训练损失和准确率
        training_loss = sum(losses) / len(losses)
        all_trg = torch.stack(all_trg, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
        training_wer = sum(all_wer) / len(all_wer) if len(all_wer) > 0 else 0
        
        print(f"\nAverage Training Loss of Epoch {epoch+1}: {training_loss:.6f} | Acc: {training_acc*100:.2f}% | WER: {training_wer:.2f}%")
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"epoch {epoch+1} | Loss {training_loss:.6f} | Acc {training_acc*100:.2f}% | WER {training_wer:.2f}%\n")
        
        # 验证模型
        val_loss, val_acc, val_wer = evaluate(model, val_loader, criterion, device)
        print(f"Average Validation Loss of Epoch {epoch+1}: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%")
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"Validation Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%\n")
        
        # 写入TensorBoard
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/wer', val_wer, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} with validation accuracy: {best_val_acc:.2f}%")
        
        # 保存每个epoch的模型
        epoch_model_path = os.path.join(args.model_path, f'continuous_slr_epoch{epoch+1:03d}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Epoch {epoch+1} model saved: {epoch_model_path}")

def evaluate(model, data_loader, criterion, device):
    """
    评估模型（与原始训练脚本保持一致）
    """
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []
    
    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(data_loader):
            imgs = imgs.to(device)
            target = target.to(device)
            
            # 模型前向传播（验证时不使用教师强制）
            outputs = model(imgs, target, teacher_forcing_ratio=0)
            
            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[1:].view(-1, output_dim)
            target_flat = target.permute(1, 0)[1:].reshape(-1)
            
            # 计算损失
            loss = criterion(outputs_flat, target_flat)
            losses.append(loss.item())
            
            # 计算准确率
            prediction = torch.max(outputs_flat, 1)[1]
            all_trg.extend(target_flat)
            all_pred.extend(prediction)
            
            # 计算WER
            batch_size = imgs.shape[0]
            prediction_seq = prediction.view(-1, batch_size).permute(1, 0).tolist()
            target_seq = target_flat.view(-1, batch_size).permute(1, 0).tolist()
            wers = []
            for i in range(batch_size):
                # add mask(remove padding, sos, eos)
                wers.append(calculate_wer(prediction_seq[i], target_seq[i]))
            all_wer.extend(wers)
    
    # 计算平均损失和准确率
    val_loss = sum(losses) / len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    val_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    val_wer = sum(all_wer) / len(all_wer) if len(all_wer) > 0 else 0
    
    return val_loss, val_acc * 100, val_wer

def main():
    parser = argparse.ArgumentParser(description='连续手语识别训练和测试 - 优化版v4')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', 
                       help='运行模式：train（训练）或 test（测试）')
    parser.add_argument('--data_path', type=str, required=True,
                       help='连续手语数据路径')
    parser.add_argument('--dict_path', type=str, default='dictionary.txt',
                       help='字典文件路径')
    parser.add_argument('--corpus_path', type=str, default='corpus.txt',
                       help='语料库文件路径')
    parser.add_argument('--model_path', type=str, default='./models/continuous_v4',
                       help='模型保存路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='测试时使用的模型检查点路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--sample_size', type=int, default=128,
                       help='图像尺寸')
    parser.add_argument('--sample_duration', type=int, default=48,
                       help='视频帧数')
    parser.add_argument('--enc_hid_dim', type=int, default=512,
                       help='编码器隐藏维度')
    parser.add_argument('--dec_hid_dim', type=int, default=512,
                       help='解码器隐藏维度')
    parser.add_argument('--emb_dim', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    parser.add_argument('--clip', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='日志间隔')
    parser.add_argument('--gpu_id', type=str, default='0',
                       help='使用的GPU ID')
    parser.add_argument('--use_char_level', action='store_true',
                       help='使用字符级别而不是词级别')
    parser.add_argument('--arch', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet架构')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                       help='教师强制比例')
    
    args = parser.parse_args()
    
    # 设置GPU
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f'使用设备: cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('使用设备: cpu')
    
    # 创建模型保存目录
    os.makedirs(args.model_path, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join('./runs', f'continuous_slr_v4_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    
    # 定义图像变换（与原始模型保持一致）
    transform = get_data_transform(args.sample_size)
    
    # 加载数据集
    print('加载数据集...')
    if args.use_char_level:
        train_dataset = CSL_Continuous_Char(args.data_path, args.corpus_path, frames=args.sample_duration, train=True, transform=transform)
        val_dataset = CSL_Continuous_Char(args.data_path, args.corpus_path, frames=args.sample_duration, train=False, transform=transform)
    else:
        train_dataset = CSL_Continuous(args.data_path, args.dict_path, args.corpus_path, frames=args.sample_duration, train=True, transform=transform)
        val_dataset = CSL_Continuous(args.data_path, args.dict_path, args.corpus_path, frames=args.sample_duration, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'数据集样本数: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}')
    
    # 计算输出维度
    output_dim = train_dataset.output_dim
    print(f'输出维度: {output_dim}')
    
    # 初始化模型（与原始模型结构相同）
    encoder = Encoder(lstm_hidden_size=args.enc_hid_dim, arch=args.arch)
    decoder = Decoder(output_dim=output_dim, emb_dim=args.emb_dim, enc_hid_dim=args.enc_hid_dim, dec_hid_dim=args.dec_hid_dim, dropout=args.dropout)
    model = Seq2Seq_v4(encoder=encoder, decoder=decoder, device=device).to(device)
    
    # 打印模型信息
    print(f'\n模型配置:')
    print(f'  ResNet架构: {args.arch}')
    print(f'  编码器隐藏维度: {args.enc_hid_dim}')
    print(f'  解码器隐藏维度: {args.dec_hid_dim}')
    print(f'  嵌入维度: {args.emb_dim}')
    print(f'  Dropout: {args.dropout}')
    print(f'  教师强制比例: {args.teacher_forcing_ratio}')
    print(f'  上下文向量: 平均值（与原始模型相同）')
    print(f'  数据变换: 标准（与原始模型相同）')
    
    # 加载预训练模型
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f'加载预训练模型: {args.checkpoint}')
    
    # 定义损失函数（与原始脚本相同）
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 创建日志文件
    log_file = os.path.join(args.model_path, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    # 开始训练
    if args.mode == 'train':
        print('\n' + '='*50)
        print('开始训练连续手语识别模型...')
        print('='*50)
        print(f'使用优化版Seq2Seq_v4模型')
        print(f'与原始模型结构相同，只做必要的优化')
        print('='*50 + '\n')
        
        train_model(args, model, train_loader, val_loader, criterion, optimizer, device, writer, log_file, train_dataset)
    elif args.mode == 'test':
        print('\n' + '='*50)
        print('开始测试连续手语识别模型...')
        print('='*50 + '\n')
        val_loss, val_acc, val_wer = evaluate(model, val_loader, criterion, device)
        print(f'Test Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%')
    
    # 关闭TensorBoard
    writer.close()

if __name__ == '__main__':
    main()
