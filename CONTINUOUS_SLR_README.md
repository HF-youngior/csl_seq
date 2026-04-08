# 连续手语识别使用指南

## 概述

这个项目专门用于连续手语识别，基于序列到序列（Seq2Seq）模型，支持词级别和字符级别的手语识别。

## 文件说明

### 核心文件
- `continuous_sign_language_recognition.py` - 主程序，包含训练和测试功能
- `config.py` - 配置文件，管理路径和参数
- `run_continuous_slr.py` - 简化的运行脚本
- `corpus.txt` - 语料库文件示例
- `dictionary.txt` - 字典文件（已存在）

### 依赖文件
- `dataset.py` - 数据集处理
- `models/Seq2Seq.py` - Seq2Seq模型实现
- `train.py` - 训练函数
- `validation.py` - 验证函数
- `tools.py` - 工具函数

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision tensorboard scikit-learn matplotlib opencv-python

# 确保所有必要文件存在
python run_continuous_slr.py
```

### 2. 数据准备

您需要准备以下数据：

1. **连续手语视频数据**：按以下结构组织
   ```
   SLR_dataset1/color/
   ├── 000000/  # 句子ID
   │   ├── 000001/  # 手语者ID
   │   │   ├── 000001/  # 重复次数
   │   │   │   ├── 000001.avi
   │   │   │   ├── 000002.avi
   │   │   │   └── ...
   │   │   └── ...
   │   └── ...
   └── ...
   ```

2. **字典文件** (`dictionary.txt`)：已提供
3. **语料库文件** (`corpus.txt`)：已提供示例

### 3. 修改配置

编辑 `config.py` 文件，修改数据路径：

```python
# 修改数据路径
config.update_data_path('continuous_data', '/root/autodl-tmp/SLR_dataset1/color')
config.update_data_path('dictionary', 'F:/SLR/dictionary.txt')
config.update_data_path('corpus', 'F:/SLR/corpus.txt')
```

### 4. 运行训练

#### 方法1：使用简化脚本
```bash
python run_continuous_slr.py
```

#### 方法2：直接使用主程序
```bash
# 训练（词级别）
python continuous_sign_language_recognition.py \
    --mode train \
    --data_path /path/to/CSL_Continuous/color \
    --dict_path dictionary.txt \
    --corpus_path corpus.txt \
    --epochs 100 \
    --batch_size 8

# 训练（字符级别）
python continuous_sign_language_recognition.py \
    --mode train \
    --data_path /path/to/CSL_Continuous/color \
    --dict_path dictionary.txt \
    --corpus_path corpus.txt \
    --use_char_level \
    --epochs 100 \
    --batch_size 8
```

### 5. 运行测试

```bash
# 测试模型
python continuous_sign_language_recognition.py \
    --mode test \
    --data_path /path/to/CSL_Continuous/color \
    --dict_path dictionary.txt \
    --corpus_path corpus.txt \
    --checkpoint ./models/continuous/best_model.pth
```

## 参数说明

### 训练参数
- `--epochs`: 训练轮数（默认100）
- `--batch_size`: 批次大小（默认8）
- `--learning_rate`: 学习率（默认1e-4）
- `--sample_duration`: 视频帧数（默认48）
- `--enc_hid_dim`: 编码器隐藏维度（默认512）
- `--dec_hid_dim`: 解码器隐藏维度（默认512）

### 模型参数
- `--use_char_level`: 使用字符级别（默认False，使用词级别）
- `--gpu_id`: GPU ID（默认'0'）

## 输出说明

### 训练输出
- 模型文件保存在 `./models/continuous/` 目录
- 训练日志保存在 `./log/` 目录
- TensorBoard日志保存在 `./runs/` 目录

### 性能指标
- **WER (Word Error Rate)**: 词错误率，越低越好
- **Accuracy**: 准确率，越高越好
- **Loss**: 损失值，越低越好

## 使用示例

### 完整训练流程
```bash
# 1. 检查环境
python run_continuous_slr.py

# 2. 训练模型
python continuous_sign_language_recognition.py \
    --mode train \
    --data_path /your/data/path \
    --epochs 50 \
    --batch_size 4

# 3. 测试模型
python continuous_sign_language_recognition.py \
    --mode test \
    --data_path /your/data/path \
    --checkpoint ./models/continuous/continuous_slr_epoch050.pth
```

### 查看训练过程
```bash
# 启动TensorBoard
tensorboard --logdir runs/

# 在浏览器中打开 http://localhost:6006
```

## 注意事项

1. **数据格式**：确保视频数据按指定结构组织
2. **GPU内存**：根据GPU内存调整batch_size
3. **训练时间**：完整训练可能需要数小时到数天
4. **模型保存**：定期检查模型保存路径的磁盘空间

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 减少sample_duration

2. **数据路径错误**
   - 检查数据路径是否正确
   - 确保数据格式符合要求

3. **模型加载失败**
   - 检查模型文件是否存在
   - 确保模型架构参数一致

### 获取帮助
如果遇到问题，请检查：
1. 错误日志文件
2. TensorBoard训练曲线
3. 数据格式是否正确
