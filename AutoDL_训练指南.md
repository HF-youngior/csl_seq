# AutoDL 中文手语识别模型训练指南

## 一、AutoDL 实例配置选择

### 推荐配置
- **GPU 类型**：至少需要 6GB 显存，推荐使用 Tesla T4 或 RTX 3060 及以上
- **内存**：至少 16GB，推荐 32GB
- **CPU**：至少 4 核，推荐 8 核
- **存储**：至少 100GB（数据集 55GB + 代码和模型 20GB + 系统和缓存 25GB）
- **网络**：选择网络较好的节点，便于数据集上传

### 实例选择理由
- 模型使用 ResNet18 作为编码器，需要足够的 GPU 显存
- 视频处理和批量训练需要较大的内存
- 多线程数据加载需要较多的 CPU 核心

## 二、数据集准备

### 数据集上传
1. **使用阿里云 OSS 上传**：
   - 将数据集压缩包上传到阿里云 OSS
   - 在 AutoDL 实例中使用 `ossutil` 工具下载到实例中

2. **数据集存储路径**：
   - 推荐存储在 `/root/autodl-tmp/` 目录下
   - 解压后路径示例：`/root/autodl-tmp/SLR_dataset/color`

### 数据集目录结构
数据集文件夹应包含以下内容：
```
SLR_dataset/
└── color/
    ├── 000000/  # 句子ID文件夹
    │   ├── P01_s1_00_0_color.avi  # 手语者1的视频
    │   ├── P02_s1_00_0_color.avi  # 手语者2的视频
    │   └── ...  # 其他手语者的视频
    ├── 000001/  # 另一个句子ID文件夹
    │   └── ...
    └── ...  # 更多句子ID文件夹
```

## 三、环境配置

### 1. 安装依赖
在 AutoDL 实例中运行以下命令：
```bash
# 安装 PyTorch 和 torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install opencv-python Pillow numpy matplotlib scikit-learn tensorboard
```

### 2. 上传代码
1. 将项目代码压缩为 zip 文件
2. 上传到 AutoDL 实例的 `/root/autodl-tmp/` 目录
3. 解压代码：
   ```bash
   unzip SLR.zip -d /root/autodl-tmp/SLR
   cd /root/autodl-tmp/SLR
   ```

### 3. 准备字典和语料库文件
1. **字典文件** (`dictionary.txt`)：
   - 格式：`ID\t中文词汇`
   - 示例：
     ```
     0\t你好
     1\t再见
     2\t谢谢
     ```

2. **语料库文件** (`corpus.txt`)：
   - 格式：`句子ID\t中文句子`
   - 示例：
     ```
     000000\t你好
     000001\t再见
     000002\t谢谢
     ```
   - 如果语料库文件不存在，代码会自动从字典文件生成

## 四、模型训练

### 1. 运行训练脚本
在 AutoDL 实例的代码目录中运行：
```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行训练
python continuous_sign_language_recognition.py \
    --mode train \
    --data_path /root/autodl-tmp/SLR_dataset/color \
    --dict_path dictionary.txt \
    --corpus_path corpus.txt \
    --model_path ./models/continuous \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --sample_duration 48 \
    --gpu_id 0
```

### 2. 训练参数说明
- `--epochs`：训练轮数，默认 100
- `--batch_size`：批次大小，根据 GPU 显存调整
- `--learning_rate`：学习率，默认 1e-4
- `--sample_duration`：每段视频提取的帧数，默认 48
- `--model_path`：模型保存路径，默认 `./models/continuous`

### 3. 训练过程监控
- **日志文件**：保存在 `log/` 目录下
- **TensorBoard**：运行 `tensorboard --logdir runs/` 查看训练过程
- **模型保存**：每轮训练后会保存模型，最终会保存 `best_model.pth`

## 五、模型评估与使用

### 1. 测试模型
训练完成后，运行以下命令测试模型：
```bash
python continuous_sign_language_recognition.py \
    --mode test \
    --data_path /root/autodl-tmp/SLR_dataset/color \
    --dict_path dictionary.txt \
    --corpus_path corpus.txt \
    --checkpoint ./models/continuous/best_model.pth \
    --batch_size 8 \
    --gpu_id 0
```

### 2. 模型格式
- 训练完成后，模型会以 `.pth` 格式保存
- 默认保存路径：`./models/continuous/best_model.pth`
- 模型文件大小：约 40-60MB（使用 ResNet18 编码器）

### 3. 使用训练好的模型
可以使用 `translate_video.py` 脚本进行视频翻译：
```bash
python translate_video.py \
    --video_path /path/to/sign_language_video.avi \
    --model_path ./models/continuous/best_model.pth \
    --dict_path dictionary.txt \
    --corpus_path corpus.txt
```

## 六、常见问题解决

### 1. 内存不足
- 减小 `batch_size` 参数
- 增加 swap 空间：
  ```bash
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

### 2. GPU 显存不足
- 减小 `batch_size` 参数
- 减小 `sample_duration` 参数
- 选择显存更大的 GPU 实例

### 3. 数据集路径错误
- 确保数据集路径正确，包含正确的目录结构
- 确保视频文件格式为 `.avi`，文件名格式正确

### 4. 训练速度慢
- 使用更大的 batch_size
- 选择性能更好的 GPU 实例
- 启用多线程数据加载（默认已启用）

## 七、训练时间估计

- **单轮训练**：约 30-60 分钟（取决于 GPU 性能和 batch_size）
- **完整训练**：约 50-100 小时（100 轮）
- **建议**：先使用 10 轮进行测试，确认流程正确后再进行完整训练

## 八、模型部署准备

训练完成后，您可以：
1. 将 `best_model.pth` 模型文件下载到本地
2. 开发前端界面，通过 API 调用模型进行推理
3. 部署模型到服务器，提供实时翻译服务

通过以上步骤，您可以在 AutoDL 上成功训练中文手语识别模型，为您的手语翻译平台提供核心功能。