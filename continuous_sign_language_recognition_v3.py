#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续手语识别主程序 - 简化版Seq2Seq_v3
专门用于训练和测试带注意力机制的连续手语识别模型
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
from models.Seq2Seq_v3 import Encoder, Decoder, Seq2Seq_v3

def get_data_transform(sample_size, is_training=True):
    """
    获取数据变换（简化版数据增强）
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((sample_size, sample_size)),
            # 只保留随机旋转，移除水平翻转（可能改变手语含义）
            transforms.RandomRotation(degrees=5),  # 小幅旋转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((sample_size, sample_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def calculate_wer(predicted, target):
    """
    计算词错误率 (Word Error Rate)
    """
    total_wer = 0
    batch_size = predicted.shape[0]
    
    for i in range(batch_size):
        pred_seq = predicted[i]
        target_seq = target[i]
        
        # 移除填充
        pred_seq = pred_seq[pred_seq != 0]
        target_seq = target_seq[target_seq != 0]
        
        if len(target_seq) == 0:
            continue
            
        # 计算编辑距离
        wer = editdistance.eval(pred_seq, target_seq) / len(target_seq)
        total_wer += wer
    
    return total_wer / batch_size if batch_size > 0 else 0

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
        
        print(f'\nepoch {epoch+1}/{args.epochs}')
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 模型前向传播
            output = model(inputs, targets, teacher_forcing_ratio=args.teacher_forcing_ratio)
            
            # 调整输出形状
            output = output.permute(1, 2, 0)  # [batch_size, trg_vocab_size, trg_len]
            target_output = targets[:, 1:]  # 移除第一个token
            output = output[:, :, :-1]  # 移除最后一个预测
            
            # 计算损失（标准CrossEntropyLoss）
            loss = criterion(output.reshape(-1, output.shape[1]), target_output.reshape(-1))
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算准确率（排除填充标记）
            _, predicted = torch.max(output, dim=1)
            
            # 创建掩码，排除填充标记（index=0）
            mask = (target_output != 0)
            correct = ((predicted == target_output) & mask).sum().item()
            total += mask.sum().item()
            train_acc += correct
            
            # 计算WER
            batch_wer = calculate_wer(predicted.cpu().numpy(), target_output.cpu().numpy())
            train_wer += batch_wer * inputs.size(0)
            
            if (i + 1) % args.log_interval == 0:
                avg_loss = train_loss / (i + 1)
                avg_acc = (train_acc / total) * 100 if total > 0 else 0
                avg_wer = train_wer / (i + 1) / args.batch_size
                
                print(f"epoch {epoch+1} | iteration {i+1} | Loss {avg_loss:.6f} | Acc {avg_acc:.2f}% | WER {avg_wer:.2f}%")
                
                # 写入TensorBoard
                writer.add_scalar('train/loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('train/accuracy', avg_acc, epoch * len(train_loader) + i)
                writer.add_scalar('train/wer', avg_wer, epoch * len(train_loader) + i)
        
        # 学习率调度（StepLR）
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = (train_acc / total) * 100 if total > 0 else 0
        avg_train_wer = train_wer / len(train_loader) / args.batch_size
        
        print(f"\nAverage Training Loss of Epoch {epoch+1}: {avg_train_loss:.6f} | Acc: {avg_train_acc:.2f}% | WER: {avg_train_wer:.2f}% | LR: {current_lr:.6f}")
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"epoch {epoch+1} | Loss {avg_train_loss:.6f} | Acc {avg_train_acc:.2f}% | WER {avg_train_wer:.2f}%\n")
        
        # 验证模型
        val_loss, val_acc, val_wer = evaluate(model, val_loader, criterion, device, args.batch_size)
        print(f"Average Validation Loss of Epoch {epoch+1}: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%")
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"Validation Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%\n")
        
        # 写入TensorBoard
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/wer', val_wer, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} with validation accuracy: {best_val_acc:.2f}%")
        
        # 保存每个epoch的模型
        epoch_model_path = os.path.join(args.model_path, f'continuous_slr_epoch{epoch+1:03d}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Epoch {epoch+1} model saved: {epoch_model_path}")

def evaluate(model, data_loader, criterion, device, batch_size):
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
            
            # 模型前向传播（验证时不使用教师强制）
            output = model(inputs, targets, teacher_forcing_ratio=0)
            
            # 调整输出形状
            output = output.permute(1, 2, 0)  # [batch_size, trg_vocab_size, trg_len]
            target_output = targets[:, 1:]
            output = output[:, :, :-1]
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.shape[1]), target_output.reshape(-1))
            total_loss += loss.item()
            
            # 计算准确率（排除填充标记）
            _, predicted = torch.max(output, dim=1)
            
            # 创建掩码，排除填充标记（index=0）
            mask = (target_output != 0)
            correct = ((predicted == target_output) & mask).sum().item()
            total += mask.sum().item()
            total_correct += correct
            
            # 计算WER
            batch_wer = calculate_wer(predicted.cpu().numpy(), target_output.cpu().numpy())
            total_wer += batch_wer * inputs.size(0)
    
    avg_loss = total_loss / len(data_loader)
    avg_acc = (total_correct / total) * 100 if total > 0 else 0
    avg_wer = total_wer / len(data_loader) / batch_size
    
    return avg_loss, avg_acc, avg_wer

def main():
    parser = argparse.ArgumentParser(description='连续手语识别训练和测试 - 简化版v3')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', 
                       help='运行模式：train（训练）或 test（测试）')
    parser.add_argument('--data_path', type=str, required=True,
                       help='连续手语数据路径')
    parser.add_argument('--dict_path', type=str, default='dictionary.txt',
                       help='字典文件路径')
    parser.add_argument('--corpus_path', type=str, default='corpus.txt',
                       help='语料库文件路径')
    parser.add_argument('--model_path', type=str, default='./models/continuous_v3',
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
    writer = SummaryWriter(log_dir=os.path.join('./runs', f'continuous_slr_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    
    # 定义图像变换（简化数据增强）
    train_transform = get_data_transform(args.sample_size, is_training=True)
    val_transform = get_data_transform(args.sample_size, is_training=False)
    
    # 加载数据集
    print('加载数据集...')
    if args.use_char_level:
        train_dataset = CSL_Continuous_Char(args.data_path, args.corpus_path, frames=args.sample_duration, train=True, transform=train_transform)
        val_dataset = CSL_Continuous_Char(args.data_path, args.corpus_path, frames=args.sample_duration, train=False, transform=val_transform)
    else:
        train_dataset = CSL_Continuous(args.data_path, args.dict_path, args.corpus_path, frames=args.sample_duration, train=True, transform=train_transform)
        val_dataset = CSL_Continuous(args.data_path, args.dict_path, args.corpus_path, frames=args.sample_duration, train=False, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'数据集样本数: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}')
    
    # 计算输出维度
    output_dim = train_dataset.output_dim
    print(f'输出维度: {output_dim}')
    
    # 初始化模型
    encoder = Encoder(lstm_hidden_size=args.enc_hid_dim, arch=args.arch)
    decoder = Decoder(output_dim=output_dim, emb_dim=args.emb_dim, enc_hid_dim=args.enc_hid_dim, dec_hid_dim=args.dec_hid_dim, dropout=args.dropout)
    model = Seq2Seq_v3(encoder=encoder, decoder=decoder, device=device).to(device)
    
    # 打印模型信息
    print(f'\n模型配置:')
    print(f'  ResNet架构: {args.arch}')
    print(f'  编码器隐藏维度: {args.enc_hid_dim}')
    print(f'  解码器隐藏维度: {args.dec_hid_dim}')
    print(f'  嵌入维度: {args.emb_dim}')
    print(f'  Dropout: {args.dropout}')
    print(f'  教师强制比例: {args.teacher_forcing_ratio}')
    print(f'  注意力机制: 启用')
    print(f'  层归一化: 启用')
    print(f'  数据增强: 简化（仅随机旋转）')
    print(f'  损失函数: 标准CrossEntropyLoss')
    print(f'  学习率调度: StepLR')
    
    # 加载预训练模型
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f'加载预训练模型: {args.checkpoint}')
    
    # 定义损失函数（标准CrossEntropyLoss）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 定义学习率调度器（StepLR）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 创建日志文件
    log_file = os.path.join(args.model_path, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    # 开始训练
    if args.mode == 'train':
        print('\n' + '='*50)
        print('开始训练连续手语识别模型...')
        print('='*50)
        print(f'使用简化版Seq2Seq_v3模型')
        print(f'核心改进: 注意力机制 + 层归一化')
        print(f'简化策略: 标准损失函数 + StepLR + 简化数据增强')
        print('='*50 + '\n')
        
        train_model(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, log_file, train_dataset)
    elif args.mode == 'test':
        print('\n' + '='*50)
        print('开始测试连续手语识别模型...')
        print('='*50 + '\n')
        val_loss, val_acc, val_wer = evaluate(model, val_loader, criterion, device, args.batch_size)
        print(f'Test Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%')
    
    # 关闭TensorBoard
    writer.close()

if __name__ == '__main__':
    main()
