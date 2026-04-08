# 持续手语识别 (Continuous Sign Language Recognition - CSLR) 项目

本项目致力于通过 **Seq2Seq 模型** 作为基线，对持续手语识别任务进行深入研究和实验。我们旨在探索和优化基于生成式模型的 CSLR 性能，并进行一系列消融实验与对比分析。

## 1. 项目概览

本项目主要关注：
- 使用 `ResNet18-LSTM Seq2Seq` 模型作为核心基线，进行手语视频到字符序列的端到端生成。
- 探索 `Seq2Seq` 模型在 CSLR 任务中的优势与局限性，特别是在处理句子级顺序模式和缺乏显式对齐监督方面的表现。
- 开展多项消融实验，验证不同组件和策略对模型性能的影响。
- 与其他 CSLR 模型（特别是基于 CTC 的模型）进行对比实验，主要通过比较最终的 Word Error Rate (WER) 指标。

## 2. 核心模型与方法

### 2.1 Baseline 模型: ResNet18-LSTM Seq2Seq

我们的基线模型是一个典型的 Encoder-Decoder 架构：
- **Encoder**: 使用 `ResNet18` 提取逐帧视觉特征，随后通过一个单层 `LSTM` 对整个视频序列进行编码。
- **Decoder**: 使用一个单层 `LSTM` 进行自回归的字符序列生成，通过 `CrossEntropy` 损失和 `Teacher Forcing` 进行训练。

该模型擅长捕获句子级上下文信息和顺序模式，但可能在更严格的泛化设置（如未见过的句子模式）下，由于缺乏显式对齐约束而表现出对训练模板的记忆依赖。

### 2.2 评估指标

主要评估指标为**标准 Word Error Rate (WER)**，计算公式为 `(Substitution + Insertion + Deletion) / Number_of_Words`。

## 3. 实验设计与探索方向

### 3.1 已完成的消融实验

项目已在 `experiments/seq2seq_ablation/` 目录下实现了多个消融实验入口，涵盖：
- **Baseline**: `train_baseline_mainwer.py`
- **创新 A (MediaPipe 融合)**: `train_ablation_a_mediapipe.py` - 探索融合 MediaPipe 提取的姿态特征。
- **创新 B (CTC 辅助约束)**: `train_ablation_b_ctcaux.py` - 引入 CTC 辅助损失以提供轻量级对齐监督。
- **创新 C (Teacher Forcing 衰减)**: `train_ablation_c_tfdecay.py` - 动态调整 Teacher Forcing 策略。

这些实验利用了统一的公共模块，包括随机种子固定 (`common_seed.py`)、日志记录 (`common_logger.py`)、WER 计算 (`common_metrics.py`) 和训练验证循环 (`common_train.py`)。

### 3.2 潜在的创新点与未来方向

根据项目进展和领域研究，未来可能探索的创新点包括：
- **多线索信息融合**: 整合 MediaPipe 提取的多种关键点特征（手部、手臂、面部），以提供更丰富的视觉信息。
- **轻量级对齐监督**: 在 `Seq2Seq` 框架中引入辅助性的对齐损失（如 CTC Head 或帧-Token 一致性损失），以增强模型的对齐鲁棒性。
- **双协议评估**: 采用更严格的评估协议，区分在 `seen-sentence` 和 `unseen-sentence` 场景下的模型性能，以全面评估泛化能力。

## 4. 数据集

本项目主要使用 **CSL100 数据集**。
- 数据集包含 `.avi` 格式的视频文件。
- 标签为字符级别。
- 数据预处理通过 `prepare_csl100.py` 脚本完成，生成语料库、字典和数据集信息文件。

## 5. 环境配置与运行

### 5.1 环境要求

- Python 3.6+
- PyTorch >= 1.8.0
- 其他依赖项请参考 `requirements.txt` 文件。

### 5.2 快速开始

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **准备 CSL100 数据集**:
    - 确保 CSL100 数据集（`.avi` 格式）已按指定结构放置。
    - 运行数据准备脚本：
      ```bash
      python prepare_csl100.py
      ```
3.  **运行实验**:
    - 参考 `Seq2Seq_训练记录与下一步计划_2026-04-08.md` 中提供的命令，运行基线或消融实验。
    - 示例（Baseline 训练命令）：
      ```bash
      python -u experiments/seq2seq_ablation/train_baseline_mainwer.py \
        --data_path /root/autodl-tmp/SLR_dataset/color \
        --corpus_path ./corpus.txt \
        --dict_path ./dictionary.txt \
        --vac_root ./VAC_CSLR-main \
        --epochs 20 \
        --batch_size 32 \
        --num_workers 6 \
        --learning_rate 1e-4 \
        --sample_size 128 \
        --sample_duration 32 \
        --enc_hid_dim 512 \
        --dec_hid_dim 512 \
        --emb_dim 256 \
        --dropout 0.5 \
        --clip 1.0 \
        --gpu_id 0 \
        --random_seed 42 \
        --model_path ./models/ablation_baseline_mainwer_seed42
      ```

## 6. 日志与结果

- **文本日志**: `log/<exp_name>_<time>.log`
- **TensorBoard**: `runs/<exp_name>_<time>/`
- **实验汇总**: `results/ablation_summary.csv`（自动记录最终测试集 WER）

## 7. 参考文献

- [Visual Alignment Constraint for Continuous Sign Language Recognition (VAC), ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf)
- [Fully Convolutional Networks for Continuous Sign Language Recognition (FCN), ECCV 2020](https://researchportal.hkust.edu.cn/en/publications/fully-convolutional-networks-for-continuous-sign-language-recogni/)
- [Iterative Alignment Network for Continuous Sign Language Recognition (Align-iOpt), CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.html)
- [Hierarchical LSTM for Sign Language Translation (HLSTM), AAAI 2018](https://aaai.org/papers/12235-hierarchical-lstm-for-sign-language-translation/)
- [Video-based Sign Language Recognition without Temporal Segmentation (LS-HAN), AAAI 2018 / arXiv](https://arxiv.org/abs/1801.10111)
- [SF-Net: Structured Feature Network for Continuous Sign Language Recognition, 2019](https://openreview.net/forum?id=N3CAox6VvK)
- [Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition (STMC), AAAI 2020](https://papers.cool/arxiv/2002.03187)
- [Continuous Sign Language Recognition through a Context-Aware Generative Adversarial Network (SLRGAN), Sensors 2021](https://www.mdpi.com/1424-8220/21/7/2437)