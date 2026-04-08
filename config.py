#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续手语识别配置文件
"""

import os

class Config:
    """配置类"""
    
    # 数据路径配置
    DATA_PATHS = {
        # 连续手语数据路径（需要根据实际情况修改）
        'continuous_data': './root/autodl-tmp/SLR_dataset1/color',
        
        # 字典文件路径
        'dictionary': 'dictionary.txt',
        
        # 语料库文件路径
        'corpus': 'corpus.txt',
        
        # 模型保存路径
        'model_save': './models/continuous',
        
        # 日志路径
        'log_dir': './log',
        
        # TensorBoard日志路径
        'tensorboard_dir': './runs'
    }
    
    # 训练参数
    TRAIN_PARAMS = {
        'epochs': 10,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'sample_size': 128,
        'sample_duration': 48,
        'enc_hid_dim': 512,
        'dec_hid_dim': 512,
        'emb_dim': 256,
        'dropout': 0.5,
        'clip': 1.0,
        'log_interval': 100,
        'gpu_id': '0'
    }
    
    # 模型参数
    MODEL_PARAMS = {
        'encoder_arch': 'resnet18',  # 可选: resnet18, resnet34, resnet50, resnet101, resnet152
        'use_char_level': False,  # True: 字符级别, False: 词级别
        'use_attention': False,  # 是否使用注意力机制
    }
    
    @classmethod
    def update_data_path(cls, key, path):
        """更新数据路径"""
        if key in cls.DATA_PATHS:
            cls.DATA_PATHS[key] = path
        else:
            raise ValueError(f"未知的路径键: {key}")
    
    @classmethod
    def update_train_param(cls, key, value):
        """更新训练参数"""
        if key in cls.TRAIN_PARAMS:
            cls.TRAIN_PARAMS[key] = value
        else:
            raise ValueError(f"未知的训练参数键: {key}")
    
    @classmethod
    def update_model_param(cls, key, value):
        """更新模型参数"""
        if key in cls.MODEL_PARAMS:
            cls.MODEL_PARAMS[key] = value
        else:
            raise ValueError(f"未知的模型参数键: {key}")
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        dirs = [
            cls.DATA_PATHS['model_save'],
            cls.DATA_PATHS['log_dir'],
            cls.DATA_PATHS['tensorboard_dir']
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")

# 示例：如何修改配置
def setup_your_config():
    """设置您的配置"""
    config = Config()
    
    # 修改数据路径（请根据您的实际路径修改）
    config.update_data_path('continuous_data', './root/autodl-tmp/SLR_dataset1/color')
    config.update_data_path('dictionary', 'F:/SLR/dictionary.txt')
    config.update_data_path('corpus', 'F:/SLR/corpus.txt')
    
    # 修改训练参数（可选）
    config.update_train_param('epochs', 50)
    config.update_train_param('batch_size', 16)
    config.update_train_param('learning_rate', 5e-5)
    
    # 修改模型参数（可选）
    config.update_model_param('use_char_level', True)
    config.update_model_param('encoder_arch', 'resnet34')
    
    return config

if __name__ == '__main__':
    # 创建目录
    Config.create_directories()
    print("配置完成！")
