# Wireless AI Experiment Suite

[中文](README.md) | [English](README.en.md)

本项目研究合成多径信道上的 CSI 重构、有限量化反馈和单用户波束预测，提供可切换的神经网络基线、固定稀疏反馈对照，以及统一的训练划分和评估接口。

## 研究问题

研究分为三个相互关联的实验方向：

1. **CSI 表示与重构**：在相同输入布局与潜变量维度下，卷积、循环与注意力编码器如何影响重构误差和空间方向保持？
2. **有限反馈重构**：在相同标称反馈预算下，4-bit 量化自编码器与训练集拟合的固定稀疏反馈，分别表现出怎样的重构误差？
3. **单用户波束预测**：不同网络能否学习从复信道到 MRT 波束的映射，并保留较高的归一化波束增益？

这些方向目前通过独立实验实现。波束模型使用真实合成信道的一个子带作为输入；重构后的 CSI 尚未串接为波束模型的输入。

## 方法概览

```text
相关多径 MISO 信道 → 独立训练 / 验证 / 测试样本
                    ├─ 连续潜变量编码 → CSI 重构 → NMSE / SGCS
                    ├─ 4-bit 潜变量量化 → CSI 重构 → 反馈预算 / NMSE
                    ├─ 固定位置稀疏反馈 → CSI 重构 → 反馈预算 / NMSE
                    └─ 单子带信道 → 波束预测 → 归一化波束增益
```

信道生成器包含指数空间相关、多径时延与功率分布、可选 Rician 分量和阴影衰落。CSI 编码器覆盖 CNN、残差 CNN、LSTM、GRU、RNN、TCN 和 Transformer；波束预测覆盖 MLP、残差 MLP 及相应序列模型。NumPy 实现提供手动反向传播与 Adam，供梯度检查和实现对照。

量化重构采用 16 级实值潜变量，训练时使用直通梯度；稀疏对照仅在训练集估计均值、固定系数位置和量化尺度。基础架构与算法来源逐项列在 [REFERENCES.md](REFERENCES.md)。项目工作集中于无线任务建模、实验接口与模型集成，不将基础网络架构作为新方法提出。

## 实验设计

| 实验 | 输入与表示 | 控制变量 | 测试指标 |
| --- | --- | --- | --- |
| 连续 CSI 重构 | 12 子带 × 32 天线，实虚双通道 | 输入布局、潜变量维度、数据种子、训练预算 | NMSE、复数 SGCS |
| 量化 / 稀疏反馈 | 32 子带 × 32 天线 | 数据种子、相同标称反馈预算、训练集拟合 | 反馈 bit 数、NMSE、复数 SGCS |
| 波束预测 | 第一个子带的 32 天线信道 | 输入样本、训练预算、MRT 监督目标 | 归一化波束增益 |

训练、验证与测试样本来自独立随机流。神经模型在训练集更新参数，以验证集 MSE 选择 epoch，恢复对应参数后计算测试指标。稀疏对照只在训练集拟合；评估代码也提供配对 bootstrap，用于对相同测试样本上的方法进行统计比较。

连续潜变量维度不是反馈 bit 数。量化模型每个实值潜变量计 4 bit，稀疏对照每个复系数计 8 bit，并假设固定均值、位置和尺度已离线共享。两组 CSI 实验的输入尺寸不同，不能直接跨组比较指标。完整数值定义、预算假设和数据布局见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `wireless_ai/channel.py` | 相关多径信道生成 |
| `wireless_ai/csi_models.py` | CSI 编码 / 解码与架构选择 |
| `wireless_ai/beam_models.py` | 单用户波束预测与架构选择 |
| `wireless_ai/quantized_csi.py` | 4-bit 量化 CNN 自编码器 |
| `wireless_ai/sparse_feedback.py` | 固定稀疏反馈与重构 |
| `wireless_ai/numpy_csi.py`, `numpy_beam.py` | 手写梯度、Adam 与参数保存 / 加载 |
| `wireless_ai/workflow.py` | 合成实验、验证集选模与测试评估入口 |
| `wireless_ai/validation/` | 指标、bootstrap、HT 布局转换和 MAT 元数据检查 |
| `tests/` | 模型前反向、梯度、量化预算与数值约定测试 |

## 安装与运行

使用 Python 3.9，在仓库目录运行：

```powershell
py -3.9 -m pip install -r requirements.txt
py -3.9 -B -X utf8 -m unittest discover -s tests -v
```

先以默认的小规模 CPU 实验检查运行环境：

```powershell
py -3.9 -B -X utf8 -m wireless_ai --task csi --arch gru --latent-dim 16
py -3.9 -B -X utf8 -m wireless_ai --task beam --arch mlp
py -3.9 -B -X utf8 -m wireless_ai --task quantized --feedback-bits 128
py -3.9 -B -X utf8 -m wireless_ai --task sparse --feedback-bits 128
```

NumPy 对照入口为 `--task csi-numpy` 和 `--task beam-numpy`。通过 `--train-samples`、`--val-samples`、`--test-samples`、`--epochs` 和 `--seed` 配置实验；`--epochs` 仅影响神经模型。`--out` 可将指标保存到一个新 JSON 文件，已有文件不会覆盖。运行结果和权重不随仓库分发。

## 当前研究状态

已实现上述模型和实验接口，35 项合成测试覆盖模型前反向、NumPy 梯度、量化与反馈预算、指标定义和输出安全；6 个任务入口已完成小规模 CPU 检查。这些检查验证实现行为，当前未提供模型收敛、跨场景泛化或方法优越性的正式比较结果。

当前信道为未校准的简化统计模型。实验范围是合成频域 CSI 与单用户波束对齐，不含真实射频采集、硬件部署或完整协议链路。进一步性能结论需要更充分的训练、多个随机种子与独立场景评估。

## 引用与许可

CSI 反馈任务参考 [CsiNet 论文](https://ieeexplore.ieee.org/document/8322184/)及[作者代码](https://github.com/sydney222/Python_CsiNet)；MRT、网络架构、量化与统计方法的完整引用见 [REFERENCES.md](REFERENCES.md)。各模型为本地任务实现与适配，使用相关方法时请引用对应原工作。

本仓库尚未指定统一源码许可证；第三方依赖遵循各自许可，引用不构成第三方代码的再分发授权。来源与依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
