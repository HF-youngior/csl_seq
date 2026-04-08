#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练结果并生成可视化图表
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_training_log(log_file):
    """
    Parse training log file
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
    
    # 提取训练数据（匹配每行开头的 epoch X | Loss ... 格式）
    train_pattern = r'^epoch\s+(\d+)\s+\|\s+Loss\s+([\d.]+)\s+\|\s+Acc\s+([\d.]+)%\s+\|\s+WER\s+([\d.]+)%'
    train_matches = re.findall(train_pattern, content, re.MULTILINE)
    
    for match in train_matches:
        epoch, loss, acc, wer = match
        epochs.append(int(epoch))
        train_losses.append(float(loss))
        train_accs.append(float(acc))
        train_wers.append(float(wer))
    
    # 提取验证数据（匹配 Validation Loss: ... 格式）
    val_pattern = r'^\s*Validation Loss:\s+([\d.]+)\s+\|\s+Acc:\s+([\d.]+)%\s+\|\s+WER:\s+([\d.]+)%'
    val_matches = re.findall(val_pattern, content, re.MULTILINE)
    
    for match in val_matches:
        loss, acc, wer = match
        val_losses.append(float(loss))
        val_accs.append(float(acc))
        val_wers.append(float(wer))
    
    return epochs, train_losses, train_accs, train_wers, val_losses, val_accs, val_wers

def plot_training_results(epochs, train_losses, train_accs, train_wers, 
                        val_losses, val_accs, val_wers, output_dir):
    """
    Plot training results charts
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss Curve
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Curve
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, val_accs, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. WER Curve
    ax3 = axes[1, 0]
    ax3.plot(epochs, train_wers, 'b-o', label='Train WER', linewidth=2, markersize=6)
    ax3.plot(epochs, val_wers, 'r-s', label='Val WER', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('WER (%)', fontsize=12)
    ax3.set_title('Training and Validation WER', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Comparison
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(epochs, val_accs, 'g-o', label='Val Accuracy', linewidth=2, markersize=6)
    line2 = ax4_twin.plot(epochs, val_wers, 'm-s', label='Val WER', linewidth=2, markersize=6)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12, color='g')
    ax4_twin.set_ylabel('WER (%)', fontsize=12, color='m')
    ax4.set_title('Validation Accuracy vs WER', fontsize=14)
    
    # Merge legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=10, loc='best')
    
    ax4.tick_params(axis='y', labelcolor='g')
    ax4_twin.tick_params(axis='y', labelcolor='m')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'training_results_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Chart saved: {output_file}')
    
    # Show chart
    plt.show()
    
    # Print statistics
    print('\n' + '='*60)
    print('Training Statistics')
    print('='*60)
    print(f'Total Epochs: {len(epochs)}')
    print(f'Final Train Loss: {train_losses[-1]:.6f}')
    print(f'Final Train Accuracy: {train_accs[-1]:.2f}%')
    print(f'Final Train WER: {train_wers[-1]:.2f}%')
    print(f'Final Val Loss: {val_losses[-1]:.6f}')
    print(f'Final Val Accuracy: {val_accs[-1]:.2f}%')
    print(f'Final Val WER: {val_wers[-1]:.2f}%')
    print(f'Best Val Accuracy: {max(val_accs):.2f}% (Epoch {val_accs.index(max(val_accs)) + 1})')
    print(f'Lowest Val WER: {min(val_wers):.2f}% (Epoch {val_wers.index(min(val_wers)) + 1})')
    print('='*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training results and generate visualization charts')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for charts')
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f'Error: Cannot find log file {args.log_file}')
        return
    
    # Parse log
    print(f'Parsing log file: {args.log_file}')
    epochs, train_losses, train_accs, train_wers, val_losses, val_accs, val_wers = parse_training_log(args.log_file)
    
    # Check data
    if len(epochs) == 0:
        print('Error: No training data found in log file')
        return
    
    print(f'Found {len(epochs)} epochs of training data')
    
    # Plot charts
    plot_training_results(epochs, train_losses, train_accs, train_wers,
                        val_losses, val_accs, val_wers, args.output_dir)

if __name__ == '__main__':
    main()
