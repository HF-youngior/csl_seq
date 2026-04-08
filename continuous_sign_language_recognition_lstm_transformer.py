#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续手语识别 - LSTM+Transformer混合模型
"""

import os
import sys
import argparse
import datetime
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# 添加模型路径
sys.path.append('./models')
from LSTM_Transformer_SL import LSTMTransformerSL
from dataset import CSL_Continuous, CSL_Continuous_Char

# 计算词错误率 (Word Error Rate)
def calculate_wer(predicted, target):
    """
    计算词错误率
    """
    total_wer = 0
    batch_size = predicted.shape[0]
    
    for i in range(batch_size):
        pred_seq = predicted[i]
        target_seq = target[i]
        
        # 移除填充
        pred_seq = pred_seq[pred_seq != 0]
        target_seq = target_seq[target_seq != 0]
        
        # 计算编辑距离
        import editdistance
        wer = editdistance.eval(pred_seq, target_seq) / len(target_seq)
        total_wer += wer
    
    return total_wer / batch_size

# 训练模型
def train_model(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, log_file, train_dataset):
    """
    训练模型
    """
    best_val_acc = 0.0
    best_model_path = os.path.join(args.model_path, 'best_model.pth')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0
        train_wer = 0
        total = 0
        
        print(f"\nepoch {epoch+1}/{args.epochs}")
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 构建目标输入（移除最后一个令牌）和目标输出（移除第一个令牌）
            tgt_input = targets[:, :-1]
            tgt_output = targets[:, 1:]
            
            output = model(inputs, tgt_input)
            
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(output, dim=2)
            correct = (predicted == tgt_output).sum().item()
            total += tgt_output.numel()
            train_acc += correct
            
            # 计算WER
            batch_wer = calculate_wer(predicted.cpu().numpy(), tgt_output.cpu().numpy())
            train_wer += batch_wer * inputs.size(0)
            
            if (i + 1) % args.log_interval == 0:
                avg_loss = train_loss / (i + 1)
                avg_acc = (train_acc / total) * 100
                avg_wer = train_wer / (i + 1) / args.batch_size
                
                print(f"iteration {i+1} | Loss {avg_loss:.6f} | Acc {avg_acc:.2f}% | WER {avg_wer:.2f}%")
                
                # 写入TensorBoard
                writer.add_scalar('train/loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('train/accuracy', avg_acc, epoch * len(train_loader) + i)
                writer.add_scalar('train/wer', avg_wer, epoch * len(train_loader) + i)
        
        # 学习率调度
        scheduler.step()
        
        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = (train_acc / total) * 100
        avg_train_wer = train_wer / len(train_loader) / args.batch_size
        
        print(f"\nAverage Training Loss of Epoch {epoch+1}: {avg_train_loss:.6f} | Acc: {avg_train_acc:.2f}% | WER: {avg_train_wer:.2f}%")
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"epoch   {epoch+1} | Loss {avg_train_loss:.6f} | Acc {avg_train_acc:.2f}% | WER {avg_train_wer:.2f}%\n")
        
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

# 评估模型
def evaluate(model, data_loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_wer = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 构建目标输入和目标输出
            tgt_input = targets[:, :-1]
            tgt_output = targets[:, 1:]
            
            output = model(inputs, tgt_input)
            
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(output, dim=2)
            correct = (predicted == tgt_output).sum().item()
            total += tgt_output.numel()
            total_correct += correct
            
            # 计算WER
            batch_wer = calculate_wer(predicted.cpu().numpy(), tgt_output.cpu().numpy())
            total_wer += batch_wer * inputs.size(0)
    
    avg_loss = total_loss / len(data_loader)
    avg_acc = (total_correct / total) * 100
    avg_wer = total_wer / len(data_loader) / data_loader.batch_size
    
    return avg_loss, avg_acc, avg_wer

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='连续手语识别 - LSTM+Transformer混合模型')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='运行模式')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--dict_path', type=str, default='dictionary.txt', help='字典文件路径')
    parser.add_argument('--corpus_path', type=str, default='corpus.txt', help='语料库文件路径')
    parser.add_argument('--model_path', type=str, default='./models/continuous_lstm_transformer', help='模型保存路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--sample_size', type=int, default=128, help='采样大小')
    parser.add_argument('--sample_duration', type=int, default=32, help='采样持续时间')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_char_level', action='store_true', help='使用字符级别的数据集')
    
    args = parser.parse_args()
    
    # 检查GPU是否可用
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型保存目录
    os.makedirs(args.model_path, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join('./runs', f'continuous_slr_lstm_transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    
    # 定义图像变换（增强版本）
    transform = transforms.Compose([
        transforms.Resize((args.sample_size, args.sample_size)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(5),       # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机颜色变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print('加载数据集...')
    if args.use_char_level:
        from dataset import CSL_Continuous_Char
        train_dataset = CSL_Continuous_Char(args.data_path, args.corpus_path, frames=args.sample_duration, train=True, transform=transform)
        val_dataset = CSL_Continuous_Char(args.data_path, args.corpus_path, frames=args.sample_duration, train=False, transform=transform)
    else:
        train_dataset = CSL_Continuous(args.data_path, args.dict_path, args.corpus_path, frames=args.sample_duration, train=True, transform=transform)
        val_dataset = CSL_Continuous(args.data_path, args.dict_path, args.corpus_path, frames=args.sample_duration, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f'数据集样本数: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}')
    
    # 计算输出维度
    output_dim = train_dataset.output_dim
    print(f'输出维度: {output_dim}')
    
    # 初始化模型
    model = LSTMTransformerSL(
        num_classes=output_dim,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 加载预训练模型
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f'加载预训练模型: {args.checkpoint}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充令牌
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 创建日志文件
    log_file = os.path.join(args.model_path, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    # 开始训练
    if args.mode == 'train':
        print('开始训练连续手语识别模型...')
        train_model(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, log_file, train_dataset)
    elif args.mode == 'test':
        print('开始测试连续手语识别模型...')
        val_loss, val_acc, val_wer = evaluate(model, val_loader, criterion, device)
        print(f'Test Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%')
    
    # 关闭TensorBoard
    writer.close()

if __name__ == '__main__':
    main()
