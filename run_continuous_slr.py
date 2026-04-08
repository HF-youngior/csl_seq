#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续手语识别运行示例
这是一个简化的运行脚本，展示如何使用连续手语识别系统
"""

import os
import sys
import subprocess

def check_requirements():
    """检查必要的文件和依赖"""
    required_files = [
        'continuous_sign_language_recognition.py',
        'dataset.py',
        'models/Seq2Seq.py',
        'train.py',
        'validation.py',
        'tools.py',
        'dictionary.txt',
        'corpus.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("错误：缺少以下必要文件：")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✓ 所有必要文件都存在")
    return True

def setup_data_paths():
    """设置数据路径"""
    print("\n=== 数据路径设置 ===")
    print("请设置您的数据路径：")
    
    # 获取用户输入
    data_path = input("连续手语数据路径 (例如: /path/to/CSL_Continuous/color): ").strip()
    
    if not data_path:
        print("使用默认路径...")
        data_path = "/path/to/your/CSL_Continuous/color"
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"警告：数据路径不存在: {data_path}")
        print("请确保路径正确后再运行训练")
    
    return data_path

def run_training(data_path):
    """运行训练"""
    print("\n=== 开始训练 ===")
    
    # 构建训练命令
    cmd = [
        'python', 'continuous_sign_language_recognition.py',
        '--mode', 'train',
        '--data_path', data_path,
        '--dict_path', 'dictionary.txt',
        '--corpus_path', 'corpus.txt',
        '--model_path', './models/continuous',
        '--epochs', '50',  # 减少训练轮数用于测试
        '--batch_size', '4',  # 减少批次大小
        '--learning_rate', '1e-4',
        '--gpu_id', '0'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("训练完成！")
        print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        print("错误输出:", e.stderr)
        return False
    
    return True

def run_testing(data_path, model_path):
    """运行测试"""
    print("\n=== 开始测试 ===")
    
    # 构建测试命令
    cmd = [
        'python', 'continuous_sign_language_recognition.py',
        '--mode', 'test',
        '--data_path', data_path,
        '--dict_path', 'dictionary.txt',
        '--corpus_path', 'corpus.txt',
        '--checkpoint', model_path,
        '--batch_size', '4',
        '--gpu_id', '0'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        # 运行测试
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("测试完成！")
        print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        print("错误输出:", e.stderr)
        return False
    
    return True

def main():
    """主函数"""
    print("连续手语识别系统")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        return
    
    # 设置数据路径
    data_path = setup_data_paths()
    
    # 选择操作
    print("\n请选择操作：")
    print("1. 训练模型")
    print("2. 测试模型")
    print("3. 训练并测试")
    print("4. 退出")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == '1':
        # 只训练
        run_training(data_path)
        
    elif choice == '2':
        # 只测试
        model_path = input("请输入模型检查点路径: ").strip()
        if not model_path:
            print("错误：需要指定模型路径")
            return
        run_testing(data_path, model_path)
        
    elif choice == '3':
        # 训练并测试
        if run_training(data_path):
            # 查找最新的模型文件
            model_dir = './models/continuous'
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    latest_model = os.path.join(model_dir, sorted(model_files)[-1])
                    print(f"使用最新模型进行测试: {latest_model}")
                    run_testing(data_path, latest_model)
                else:
                    print("未找到训练好的模型文件")
            else:
                print("模型目录不存在")
        
    elif choice == '4':
        print("退出程序")
        return
        
    else:
        print("无效选择")

if __name__ == '__main__':
    main()
