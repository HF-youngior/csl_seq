#!/usr/bin/env python3
"""
连续手语识别 - Transformer版本
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 添加模型路径
sys.path.append('./models')
from Transformer_SL import TransformerSL
from dataset import CSL_Continuous

def main():
    parser = argparse.ArgumentParser(description='连续手语识别 - Transformer版本')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='运行模式')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--dict_path', type=str, default='dictionary.txt', help='词典路径')
    parser.add_argument('--corpus_path', type=str, default='corpus.txt', help='语料库路径')
    parser.add_argument('--model_path', type=str, default='./models/continuous_transformer', help='模型保存路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--sample_size', type=int, default=128, help='采样大小')
    parser.add_argument('--sample_duration', type=int, default=32, help='采样持续时间')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--nhead', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃率')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_char_level', action='store_true', help='使用字符级别数据集')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型保存目录
    os.makedirs(args.model_path, exist_ok=True)
    
    # 创建日志目录
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'continuous_slr_transformer_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join('./runs', f'continuous_slr_transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    
    # 加载数据集
    print('加载数据集...')
    
    # 定义图像变换
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((args.sample_size, args.sample_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
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
    model = TransformerSL(
        num_classes=output_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 加载检查点
    if args.checkpoint:
        print(f'加载检查点: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    if args.mode == 'train':
        train_model(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, log_file, train_dataset)
    else:
        test_model(args, model, val_loader, criterion, device, log_file, val_dataset)
    
    writer.close()

def calculate_wer(predicted, target, eos_token=2):
    """
    计算词错误率(WER)
    """
    total_words = 0
    total_errors = 0
    
    for pred, tgt in zip(predicted, target):
        # 移除填充和结束令牌
        pred = pred[pred != 0]
        pred = pred[pred != eos_token]
        tgt = tgt[tgt != 0]
        tgt = tgt[tgt != eos_token]
        
        total_words += len(tgt)
        total_errors += sum(p != t for p, t in zip(pred, tgt)) + abs(len(pred) - len(tgt))
    
    return (total_errors / total_words) * 100 if total_words > 0 else 0

def train_model(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, log_file, train_dataset):
    print('开始训练连续手语识别模型...')
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        train_acc = 0
        train_wer = 0
        total = 0
        
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
                avg_wer = train_wer / ((i + 1) * args.batch_size)
                
                print(f'epoch   {epoch} | iteration   {i+1} | Loss {avg_loss:.6f} | Acc {avg_acc:.2f}% | WER {avg_wer:.2f}%')
                
                # 写入TensorBoard
                writer.add_scalar('Training/Loss', avg_loss, (epoch-1)*len(train_loader) + i)
                writer.add_scalar('Training/Accuracy', avg_acc, (epoch-1)*len(train_loader) + i)
                writer.add_scalar('Training/WER', avg_wer, (epoch-1)*len(train_loader) + i)
        
        # 验证
        val_loss, val_acc, val_wer = evaluate(model, val_loader, criterion, device, val_loader.dataset)
        
        print(f'Average Training Loss of Epoch {epoch}: {train_loss / len(train_loader):.6f} | Acc: {train_acc / total * 100:.2f}% | WER {train_wer / len(train_loader) / args.batch_size:.2f}%')
        print(f'Average Validation Loss of Epoch {epoch}: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%')
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f'epoch   {epoch} | Loss {train_loss / len(train_loader):.6f} | Acc {train_acc / total * 100:.2f}% | WER {train_wer / len(train_loader) / args.batch_size:.2f}%\n')
            f.write(f'Validation Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | WER: {val_wer:.2f}%\n')
        
        # 写入TensorBoard
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        writer.add_scalar('Validation/WER', val_wer, epoch)
        
        # 保存模型
        model_save_path = os.path.join(args.model_path, f'continuous_slr_transformer_epoch{epoch:03d}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Epoch {epoch} 模型已保存: {model_save_path}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_save_path = os.path.join(args.model_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f'最佳模型已更新: {best_model_save_path}')
        
        scheduler.step()

def evaluate(model, data_loader, criterion, device, dataset):
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

def test_model(args, model, test_loader, criterion, device, log_file, test_dataset):
    print('开始测试连续手语识别模型...')
    
    test_loss, test_acc, test_wer = evaluate(model, test_loader, criterion, device, test_dataset)
    
    print(f'Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.2f}% | Test WER: {test_wer:.2f}%')
    
    # 写入日志
    with open(log_file, 'a') as f:
        f.write(f'Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.2f}% | Test WER: {test_wer:.2f}%\n')

if __name__ == '__main__':
    main()
