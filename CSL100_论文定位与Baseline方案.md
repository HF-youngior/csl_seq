# CSL100 论文定位与 Baseline 方案

## 1. 结论先锁定

1. `VAC` 和你当前的 `Seq2Seq` 不是同一个主框架。
2. 它们只在最底层共享了一个宽泛思路: `RGB视频 -> 2D CNN 提帧特征`。
3. 你的本地 `Seq2Seq` 可以作为论文里的 baseline, 但建议命名为:
   - `ResNet18-LSTM Seq2Seq baseline (ours)`
4. 这个 baseline 应该被表述为“我们的工程基线”，而不是“复现某篇现成 CSL 论文”。
5. 你现在最该做的不是继续争论 VAC 为什么没到论文分数，而是先把 `Seq2Seq` 用统一的 `VAC evaluator` 跑一遍，拿到可信的主表分数。

## 2. 本地代码核对结果

### 2.1 VAC 本地实现

主文件:

- `f:\SLR\VAC_CSLR-main\slr_network.py`
- `f:\SLR\VAC_CSLR-main\modules\tconv.py`

本地 VAC 的真实结构是:

1. `ResNet18` 逐帧视觉特征提取  
   见 `slr_network.py:49-50`
2. `TemporalConv` 做时序卷积压缩  
   见 `slr_network.py:51-55`, `tconv.py:18-36`
3. `2-layer BiLSTM` 做长时序建模  
   见 `slr_network.py:63-64`
4. 分类头输出时序 logits  
   见 `slr_network.py:65-72`, `103`
5. 训练损失是 `SeqCTC + ConvCTC (+ Dist 可选)`  
   见 `slr_network.py:150-158`
6. 推理阶段走 `CTC decode`，支持 `beam` 和 `max`  
   见 `slr_network.py:104-108`, `utils\decode.py:17-56`

一句话总结:

- 这是一个 **CTC-based CSLR**。
- 它本质上在做“时序对齐识别”，不是自回归生成。

### 2.2 Seq2Seq 本地实现

主文件:

- `f:\SLR\models\Seq2Seq.py`
- `f:\SLR\continuous_sign_language_recognition.py`
- `f:\SLR\train.py`
- `f:\SLR\validation.py`

本地 Seq2Seq 的真实结构是:

1. `ResNet18` 逐帧提特征  
   见 `Seq2Seq.py:22-34`
2. `1-layer LSTM encoder` 编码整段视频  
   见 `Seq2Seq.py:35-39`, `56-62`
3. 用 encoder 输出均值做单个 `context vector`  
   见 `Seq2Seq.py:123-128`
4. `1-layer LSTM decoder` 自回归生成字符序列  
   见 `Seq2Seq.py:65-103`, `132-146`
5. 训练目标是 `CrossEntropy + teacher forcing`  
   见 `train.py:62-72`, `139-146`
6. 验证时关闭 teacher forcing，按 token 逐步解码  
   见 `validation.py:49-50`

一句话总结:

- 这是一个 **encoder-decoder 自回归序列生成模型**。
- 它更像“根据整段视频生成整句字符序列”。

### 2.3 两者到底哪里一样，哪里不一样

相同点:

1. 都吃 RGB 视频。
2. 都先做逐帧 2D CNN 提特征。
3. 你这两个本地实现都用了 `ResNet18` 作为帧级视觉骨干。

不同点:

1. `VAC` 是 `CTC` 路线，`Seq2Seq` 是 `CrossEntropy + autoregressive decoder` 路线。
2. `VAC` 显式依赖时序对齐，`Seq2Seq` 不显式做对齐。
3. `VAC` 的时序主干是 `TemporalConv + BiLSTM`，`Seq2Seq` 是 `LSTM encoder + LSTM decoder`。
4. `VAC` 解码是 `CTC beam/max`，`Seq2Seq` 解码是逐 token 递归生成。
5. `VAC` 更偏“识别”，`Seq2Seq` 更偏“句子条件生成”。

所以更准确的说法是:

- **它们不是同一个框架只改了几层。**
- **它们只是共享了帧级视觉骨干这一层。**

## 3. 你的本地 Seq2Seq 为什么可能显得特别强

这件事很关键，论文里必须写清楚。

### 3.1 旧版 Seq2Seq WER 和 VAC WER 不是同一口径

你本地旧版 Seq2Seq 的 WER 计算在:

- `f:\SLR\tools.py:121-143`

它是直接对脚本里的 token 序列做编辑距离，没有走 VAC 的:

1. `CTM` 导出
2. `STM` 对齐
3. 官方风格 `python_wer_evaluation.py`

所以:

- 旧 Seq2Seq WER 可以保留，
- 但不应该直接进论文主表，
- 主表必须用统一后的 VAC evaluator。

### 3.2 Seq2Seq 对 seen-sentence split 天然更有利

这点有文献支持。

`FCN` 论文明确指出，很多 `CNN+RNN` 混合模型在 CSL 这类设置下，容易学到已经见过的句子模式，对 unseen sequence patterns 学得不好。  
来源: HKUST 的 FCN 论文摘要写明 “most of them fail in learning unseen sequence patterns”。  

这正好解释你目前的观察:

1. 如果 split 比较容易，
2. 如果句子模板在训练里出现过，
3. 如果评估脚本又更松，

那么 `Seq2Seq` 很可能因为:

- 自回归语言建模能力，
- teacher forcing 训练，
- seen-sentence 模板记忆，

拿到一个看起来异常低的 WER。

这不代表它一定“真实泛化更强”。

## 4. VAC 论文和你本地 VAC 工程，不是严格一比一复现

这点也必须说清楚。

根据 VAC 论文:

1. `CSL` 没有官方 split，作者沿用前人 `8:2` 设置。
2. 在 `CSL` 上作者实际采用的是 `VGG11 backbone`。
3. 在 `PHOENIX14` 上作者强调的是 `ResNet18`。

因此你当前本地 VAC-CSL100 工程和论文存在三层不完全一致:

1. 数据集标签粒度:
   - 论文常见是 gloss 级
   - 你这里是中文字符级
2. backbone:
   - 论文 CSL 用 `VGG11`
   - 你本地工程目前主要用 `ResNet18`
3. split 协议:
   - 论文采用历史 `8:2`
   - 你一度还测试过更严格的独立划分

这意味着:

- 你不能把“论文 1.6 WER”直接当成你当前工程的应达目标，
- 至少不能不加条件地直接横比。

## 5. 文献池里哪些适合放主表

下面这批论文和 CSL 的关联最强，也最适合你后面写 related work。

### 5.1 HLSTM / HLSTM-attn (AAAI 2018)

论文:

- `Hierarchical LSTM for Sign Language Translation`
- AAAI 2018
- https://aaai.org/papers/12235-hierarchical-lstm-for-sign-language-translation/

核心结构:

1. 3D CNN 抽取 clip 级视觉特征
2. 分层 LSTM 建模 viseme / sequence
3. attention-aware 权重
4. encoder-decoder 风格翻译

适合怎么写:

- 它代表较早的层级递归 encoder-decoder 路线。
- 优点是句子级建模强。
- 缺点是递归路径长，优化重，并且更容易吃到 seen-sentence split 的红利。

重要点:

- Align-iOpt 论文在 CSL 上引用了 `HLSTM` / `HLSTM-attn` 在 seen-sentence 和 unseen-sentence 两种 split 下的结果。
- 其中 `Split I` 是 signer-independent but seen-sentence。
- `Split II` 是 unseen-sentence。

### 5.2 LS-HAN (AAAI 2018)

论文:

- `Video-based Sign Language Recognition without Temporal Segmentation`
- https://arxiv.org/abs/1801.10111

核心结构:

1. two-stream CNN
2. latent space
3. hierarchical attention network

适合怎么写:

- 它代表更早的“无显式分段 + latent space + hierarchical attention”路线。
- 可以批评的点是结构长、模块多、训练链路偏重。

### 5.3 Align-iOpt (CVPR 2019)

论文:

- `Iterative Alignment Network for Continuous Sign Language Recognition`
- https://openaccess.thecvf.com/content_CVPR_2019/html/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.html

核心结构:

1. `3D-ResNet` 做特征学习
2. `encoder-decoder + CTC decoder`
3. `soft-DTW` 对齐约束
4. iterative optimization

它最有价值的地方不是只有结果，而是它把 CSL split 明确拆成:

1. `Split I`: signer-independent, 句子见过
2. `Split II`: unseen-sentence

这对你论文特别重要，因为它证明:

- **不同 split 难度差异非常大**，
- 低 WER 很可能和 split 设置强相关。

### 5.4 SF-Net (2019)

论文:

- `SF-Net: Structured Feature Network for Continuous Sign Language Recognition`
- https://openreview.net/forum?id=N3CAox6VvK

核心结构:

1. 从 frame level 到 gloss level 再到 sentence level 的结构化特征编码
2. end-to-end 训练
3. 不依赖外部预训练流程

适合怎么写:

- 它是“层次结构特征建模”路线。
- 比早期 encoder-decoder 更强，但结构依旧较重。

### 5.5 FCN (ECCV 2020)

论文:

- `Fully Convolutional Networks for Continuous Sign Language Recognition`
- https://researchportal.hkust.edu.hk/en/publications/fully-convolutional-networks-for-continuous-sign-language-recogni/

核心结构:

1. CNN + fully convolutional temporal modeling
2. `GFE` 模块增强 gloss feature alignment
3. 不走传统 `CNN+RNN` 主路

它对你最有用的不是分数，而是叙事:

- 作者明确指出很多 `CNN+RNN hybrid` 模型对 unseen sequence patterns 学不好。

这个点你后面可以直接转化成论文动机:

- 纯 `Seq2Seq` 句子生成器很可能擅长记模板，
- 但缺少显式对齐监督，
- 因此我们要补对齐信息或多线索信息。

### 5.6 STMC (AAAI 2020)

论文:

- `Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition`
- https://papers.cool/arxiv/2002.03187

核心结构:

1. `SMC` 空间多线索分解
2. `TMC` 时序多线索建模
3. 同时利用手形、面部、姿态等 cue

适合怎么写:

- 它代表“多线索显式建模”路线。
- 你以后如果想接 `MediaPipe`，它会是特别好的对比对象。

### 5.7 VAC (ICCV 2021)

论文:

- `Visual Alignment Constraint for Continuous Sign Language Recognition`
- https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf

核心结构:

1. frame-wise feature extractor
2. alignment module
3. auxiliary classifier
4. `VE` / `VA` 两类额外监督

适合怎么写:

- 它代表较强的 CTC 对齐范式。
- 优点是对齐监督明确。
- 缺点是句子级生成先验弱于生成式 decoder。

### 5.8 SLRGAN (Sensors 2021)

论文:

- `Continuous Sign Language Recognition through a Context-Aware Generative Adversarial Network`
- https://www.mdpi.com/1424-8220/21/7/2437

核心结构:

1. context-aware generation
2. sentence / gloss discriminators
3. 生成式对抗建模

适合怎么写:

- 如果你未来想强调“句子级上下文”或者“生成式约束”，它是很好的一篇 related work。
- 但如果你当前数据只是单句级，没有真实对话上下文，那么它更适合放 related work，不一定适合当主复现实验对象。

## 6. 一份适合直接写到论文里的对比表述

### 6.1 你可以怎么写 VAC

`VAC` 属于典型的 CTC-based CSLR 框架，其核心思想是通过视觉特征提取、时序对齐建模和辅助分类约束，强化 frame-to-label 的显式对齐能力。与生成式 encoder-decoder 不同，VAC 更强调识别路径中的对齐监督，因此在 gloss/character alignment 学习上通常更稳定，但对句子级语言先验的利用相对有限。

### 6.2 你可以怎么写 Seq2Seq baseline

我们采用一个 `ResNet18-LSTM Seq2Seq baseline` 作为生成式基线。该模型首先利用 `ResNet18` 提取逐帧视觉特征，再通过 LSTM encoder 建模整段视频，最终由 LSTM decoder 自回归生成字符序列。与 CTC-based 方法相比，该类方法更容易捕获句子级顺序模式，但缺少显式对齐约束，因此在更严格的泛化设置下可能出现模板记忆强、对齐鲁棒性不足的问题。

## 7. 你后面论文最稳的一条主线

如果你打算围绕当前 `Seq2Seq` 做论文，最稳的路线不是继续硬拧 VAC，而是:

1. 把 `Seq2Seq` 作为主 baseline
2. 把 `VAC` 作为经典 CTC 对照
3. 在 `Seq2Seq` 上加你的创新点

我建议你后面 3 个创新点优先从这里出:

### 7.1 多线索信息

例如:

1. `MediaPipe` 手部关键点
2. 面部关键点
3. 姿态关键点

好处:

- 能直接对接 `STMC` 这条文献线，
- 论文叙事很顺，
- 消融实验也好做。

### 7.2 给 Seq2Seq 补一个轻量对齐监督

例如:

1. auxiliary CTC head
2. frame-token consistency loss
3. monotonic alignment regularizer

好处:

- 这正好能把 `VAC` 的优势借过来，
- 形成“生成式句子建模 + 显式对齐约束”的新模型动机。

### 7.3 双协议评估

即:

1. seen-sentence 口径
2. 更严格的 unseen-sentence 或独立划分口径

好处:

- 这是 reviewer 很在意的问题，
- 也能解释为什么某些方法在 easy split 下很强，但 harder split 下退化明显。

## 8. 你现在最该立刻做的实验

顺序建议固定成这样:

1. 用统一 split 重新训练 `Seq2Seq`
2. 导出 `CTM`
3. 用 `VAC evaluator` 重新计算主表 WER
4. 同时保留旧版 Seq2Seq 内部 WER 作为附录数值

主表里只放:

- 统一评估脚本下的 WER

附录里可以补:

- 旧 Seq2Seq 脚本自己的 WER

如果这两者差得很大，反而是一个很好的论文点:

- **评估协议本身会显著改变模型表面表现。**

## 9. 你现在能怎么摆论文里的 baseline 和对比对象

### 主 baseline

1. `ResNet18-LSTM Seq2Seq baseline (ours)`

### 经典对照

1. `VAC`

### 文献对比池

1. `HLSTM / HLSTM-attn`
2. `LS-HAN`
3. `Align-iOpt`
4. `SF-Net`
5. `FCN`
6. `STMC`
7. `SLRGAN`

### 建议

如果你的统一评估后 `Seq2Seq` 仍然明显强于当前 VAC 复现，那么:

1. `Seq2Seq` 完全可以当主 baseline
2. `VAC` 当经典 CTC 对照
3. 你的创新模型建立在 `Seq2Seq` 之上会更自然

## 10. 文献中最值得你直接引用的几个点

1. VAC 论文说明:
   - CSL 无官方 split
   - 采用历史 `8:2`
   - CSL 上用 `VGG11`
   - `Baseline+VAC` 在该 setting 下报 `1.6 WER`
2. FCN 论文说明:
   - 很多 `CNN+RNN hybrid` 模型对 unseen sequence patterns 学不好
3. Align-iOpt 论文说明:
   - CSL 至少需要区分 seen-sentence 的 `Split I` 和 unseen-sentence 的 `Split II`
4. STMC 说明:
   - 多线索 cue 是一条合理且有效的路线

## 11. 最后给你一个很实际的判断

如果你现在的目标是“尽快搭一个能写论文的主线”，那就不要再把时间主要砸在把 VAC 从 `50%+` 硬拧到论文 `1.6%`。

更稳的路线是:

1. 把 `Seq2Seq` 用统一 WER 跑成可信 baseline
2. 让 `VAC` 作为经典对照
3. 后续创新放在 `Seq2Seq + 多线索 + 对齐辅助`

这条线更容易:

1. 讲清楚动机
2. 做出消融
3. 解释为什么有效
4. 写成一篇完整论文

## 12. 参考链接

1. VAC ICCV 2021  
   https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf
2. FCN ECCV 2020  
   https://researchportal.hkust.edu.hk/en/publications/fully-convolutional-networks-for-continuous-sign-language-recogni/
3. Align-iOpt CVPR 2019  
   https://openaccess.thecvf.com/content_CVPR_2019/html/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.html
4. HLSTM AAAI 2018  
   https://aaai.org/papers/12235-hierarchical-lstm-for-sign-language-translation/
5. LS-HAN AAAI 2018 / arXiv  
   https://arxiv.org/abs/1801.10111
6. SF-Net arXiv / OpenReview  
   https://openreview.net/forum?id=N3CAox6VvK
7. STMC AAAI 2020  
   https://papers.cool/arxiv/2002.03187
8. SLRGAN Sensors 2021  
   https://www.mdpi.com/1424-8220/21/7/2437
