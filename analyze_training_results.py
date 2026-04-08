#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练结果并生成可视化图表
"""

import os
import re
import shutil
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file):
    """
    解析训练日志文件
    """
    epochs = []
    train_losses = []
    train_accs = []
    train_wers = []
    val_losses = []
    val_accs = []
    val_wers = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取每轮的训练数据
    epoch_pattern = r'epoch\s+(\d+)\s*\|\s*iteration\s+\d+\s*\|\s*Loss\s+([\d.]+)\s*\|\s*Acc\s+([\d.]+)%\s*\|\s*WER\s+([\d.]+)%'
    train_pattern = r'Average Training Loss of Epoch (\d+): ([\d.]+) \| Acc: ([\d.]+)% \| WER ([\d.]+)%'
    val_pattern = r'Average Validation Loss of Epoch (\d+): ([\d.]+) \| Acc: ([\d.]+)% \| WER: ([\d.]+)%'
    
    # 提取训练数据
    train_matches = re.findall(train_pattern, content)
    for match in train_matches:
        epoch = int(match[0])
        train_loss = float(match[1])
        train_acc = float(match[2])
        train_wer = float(match[3])
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_wers.append(train_wer)
    
    # 提取验证数据
    val_matches = re.findall(val_pattern, content)
    for match in val_matches:
        epoch = int(match[0])
        val_loss = float(match[1])
        val_acc = float(match[2])
        val_wer = float(match[3])
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_wers.append(val_wer)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_wers': train_wers,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_wers': val_wers
    }

def find_best_model(data, model_dir):
    """
    找出最佳模型并复制为bestmodel.pth
    """
    if not data['val_accs']:
        return None
    
    best_idx = np.argmax(data['val_accs'])
    best_epoch = data['epochs'][best_idx]
    best_val_acc = data['val_accs'][best_idx]
    best_val_loss = data['val_losses'][best_idx]
    best_val_wer = data['val_wers'][best_idx]
    
    # 查找最佳模型文件
    best_model_file = f'continuous_slr_epoch{best_epoch:03d}.pth'
    best_model_path = os.path.join(model_dir, best_model_file)
    
    # 复制最佳模型为bestmodel.pth
    if os.path.exists(best_model_path):
        best_model_copy_path = os.path.join(model_dir, 'bestmodel.pth')
        shutil.copy2(best_model_path, best_model_copy_path)
        print(f'最佳模型已复制: {best_model_path} -> {best_model_copy_path}')
    else:
        print(f'警告: 最佳模型文件不存在: {best_model_path}')
    
    return {
        'epoch': best_epoch,
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'val_wer': best_val_wer,
        'model_file': best_model_file
    }

def plot_results(data, output_dir):
    """
    绘制训练结果图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查数据长度，确保训练和验证数据匹配
    min_len = min(len(data['epochs']), len(data['train_losses']), len(data['train_accs']), len(data['train_wers']))
    val_len = min(len(data['val_losses']), len(data['val_accs']), len(data['val_wers']))
    
    # 如果验证数据比训练数据少，截断训练数据
    if val_len < min_len:
        print(f'警告: 验证数据({val_len}轮)比训练数据({min_len}轮)少，将截断训练数据')
        epochs = data['epochs'][:val_len]
        train_losses = data['train_losses'][:val_len]
        train_accs = data['train_accs'][:val_len]
        train_wers = data['train_wers'][:val_len]
        val_losses = data['val_losses']
        val_accs = data['val_accs']
        val_wers = data['val_wers']
    else:
        epochs = data['epochs']
        train_losses = data['train_losses']
        train_accs = data['train_accs']
        train_wers = data['train_wers']
        val_losses = data['val_losses']
        val_accs = data['val_accs']
        val_wers = data['val_wers']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # WER曲线
    axes[1, 0].plot(epochs, train_wers, 'b-', label='Train WER', linewidth=2)
    axes[1, 0].plot(epochs, val_wers, 'r-', label='Validation WER', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('WER (%)', fontsize=12)
    axes[1, 0].set_title('Word Error Rate (WER) Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 综合对比
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1, = ax4.plot(epochs, val_accs, 'b-', label='Validation Accuracy', linewidth=2)
    line2, = ax4_twin.plot(epochs, val_wers, 'r-', label='Validation WER', linewidth=2)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', color='b', fontsize=12)
    ax4_twin.set_ylabel('WER (%)', color='r', fontsize=12)
    ax4.set_title('Validation Metrics Comparison', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=10)
    
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'可视化结果已保存: {output_path}')
    
    plt.close()

def print_summary(data, best_model):
    """
    打印训练结果摘要
    """
    print('\n' + '='*80)
    print('训练结果摘要')
    print('='*80)
    print(f'总训练轮数: {len(data["epochs"])}')
    print(f'训练集准确率范围: {min(data["train_accs"]):.2f}% - {max(data["train_accs"]):.2f}%')
    print(f'验证集准确率范围: {min(data["val_accs"]):.2f}% - {max(data["val_accs"]):.2f}%')
    print(f'训练集WER范围: {min(data["train_wers"]):.2f}% - {max(data["train_wers"]):.2f}%')
    print(f'验证集WER范围: {min(data["val_wers"]):.2f}% - {max(data["val_wers"]):.2f}%')
    
    if best_model:
        print('\n' + '='*80)
        print('最佳模型')
        print('='*80)
        print(f'轮数: Epoch {best_model["epoch"]}')
        print(f'验证集准确率: {best_model["val_acc"]:.2f}%')
        print(f'验证集损失: {best_model["val_loss"]:.6f}')
        print(f'验证集WER: {best_model["val_wer"]:.2f}%')
        print(f'原始模型文件: {best_model["model_file"]}')
        print(f'最佳模型文件: bestmodel.pth')
    
    print('='*80 + '\n')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析训练结果并生成可视化图表')
    parser.add_argument('--log_file', type=str, required=True, help='训练日志文件路径')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--model_dir', type=str, required=True, help='模型文件所在目录')
    
    args = parser.parse_args()
    
    # 解析日志文件
    print(f'正在解析日志文件: {args.log_file}')
    data = parse_log_file(args.log_file)
    
    if not data['epochs']:
        print('错误: 无法从日志文件中提取训练数据')
        return
    
    # 找出最佳模型
    best_model = find_best_model(data, args.model_dir)
    
    # 打印摘要
    print_summary(data, best_model)
    
    # 生成可视化图表
    plot_results(data, args.output_dir)
    
    print('分析完成！')

if __name__ == '__main__':
    main()
