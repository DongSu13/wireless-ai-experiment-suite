# Wireless AI Experiment Suite

[English](README.en.md)

面向学习和数值验证的无线 AI 实验工具集：整理本地开发的合成信道、CSI 重建、单用户波束预测、量化反馈对照及验证代码。贡献主要是实验接口、模型集成、手写梯度、数值约定与测试；不声称提出这些基础模型，也不声称精确复现某篇论文或达到论文报告的性能。

目前作为私有候选仓库上传；公开发布需要另行确认。仓库只保留代码、合成测试和介绍，不包含真实数据、模型权重、内部报告或个人资料。

## 内容

| 模块 | 内容 |
| --- | --- |
| `wireless_ai/channel.py` | 相关多径 MISO 合成频域信道，可选 Rician 分量与阴影衰落 |
| `wireless_ai/csi_models.py` | CNN、残差 CNN、LSTM、GRU、RNN、TCN、Transformer 连续潜变量重建 |
| `wireless_ai/beam_models.py` | MLP、残差 MLP、LSTM、GRU、RNN、TCN、Transformer 单用户波束预测 |
| `wireless_ai/numpy_csi.py`, `numpy_beam.py` | NumPy 手动反向传播、Adam 与前向推理 |
| `wireless_ai/quantized_csi.py` | 训练和评估均使用 16 级量化的 CNN 自编码器 |
| `wireless_ai/sparse_feedback.py` | 训练集拟合的固定位置、实虚部各 4 bit 的稀疏反馈对照 |
| `wireless_ai/validation/` | NMSE、复数 SGCS、归一化波束增益、反馈预算、配对 bootstrap 和 MAT 元数据检查 |

## 运行

使用 Python 3.9。依赖见 [requirements.txt](requirements.txt)，不需要下载数据。基础 NumPy 示例：

```powershell
py -3.9 -m pip install -r requirements.txt
py -3.9 -B -X utf8 -m unittest discover -s tests -v
py -3.9 -B -X utf8 -m wireless_ai --task csi-numpy
py -3.9 -B -X utf8 -m wireless_ai --task beam-numpy
py -3.9 -B -X utf8 -m wireless_ai --task csi --arch gru --latent-dim 16
py -3.9 -B -X utf8 -m wireless_ai --task beam --arch mlp
py -3.9 -B -X utf8 -m wireless_ai --task quantized --feedback-bits 128
py -3.9 -B -X utf8 -m wireless_ai --task sparse --feedback-bits 128
```

CLI 是小规模 CPU 示例，默认 32/8/8 条训练/验证/测试样本和 2 个 epoch，仅用于检查流程。三份数据来自独立的随机流；神经模型按验证集 MSE 选择 epoch，最后报告测试集指标。CSI 与波束是两个独立任务：波束输入为真实合成信道的第一个子带，没有串接压缩重建后的反馈链路。

`--out` 可保存到仓库之外的新 JSON 文件；已有文件不会覆盖。默认不保存权重。模型类可以直接前向推理；NumPy 类提供 NPZ 参数保存/加载，仅加载可信、符合接口的文件。

## 数值约定与限制

- 信道为简化教学仿真，未按 3GPP 或 QuaDRiGa 校准。默认 `[sample,12,32]`，量化和稀疏示例使用 `[sample,32,32]`；后者在 CLI 中仍是频域子带/天线布局，不是 COST2100 的角度时延域数据。
- Rician K 指阴影衰落前的 LoS / 散射期望功率比。两者均按每根天线单位期望功率归一化。
- `latent_dim` 是连续浮点潜变量数，不能称为反馈 bit 数。量化模型按 `feedback_bits/4` 个 16 级实值潜变量计预算；稀疏模型每个复数系数计 8 bit。两者没有打包传输码流、协议头或熵编码。
- 稀疏对照的均值、位置和量化尺度仅由训练集拟合并离线共享，不计入每样本负载；跨场景更新这些信息的成本需另算。它不是 3GPP Type-II 实现。
- 信号约定 `y=hᴴwx+n`，MRT 为 `w=h/||h||`。归一化波束增益测量方向对齐，不代表吞吐量或多用户优化。
- NMSE 按样本误差功率/真实功率计算，再取 `10log10(mean(linear NMSE))`。复数 SGCS 为每根向量的平方复数余弦，最后按样本平均；调用者必须先明确域和天线轴。
- bootstrap 假设样本对可独立重采样；轨迹或时间相关数据需要分组/区块方法。零功率向量不能评分。
- CsiNet 风格 `HT` 转换工具仅支持 `[sample,2048]` 的实部/虚部布局并减去 0.5，输出 `[sample,angle,delay]`。不能直接当作频域 SGCS 输入。MAT 检查默认只输出文件名，但文件名和变量名也可能包含信息，分享前须复查。
- 合成测试不证明实际无线性能、GPU 逐位可复现或原论文复现成功。

## 来源和许可

各模型的概念来源及作者代码链接见 [REFERENCES.md](REFERENCES.md)，来源边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。这些网络是本地教学实现/集成，引用基础架构不等于声明复现其原始任务。

不打包第三方完整模型仓库、数据集或预训练权重，也不自动下载。若使用公开数据，请从对应作者页面取得并核实使用条件；非公开数据不进入本仓库。引用和“仅供学习”说明不能替代许可。目前没有为全部代码授予统一开源许可证，公开前仍需确认权利范围。
