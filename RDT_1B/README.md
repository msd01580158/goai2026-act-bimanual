# RDT-1B 大规模 VLA：部署与验证

> 本目录记录 RDT-1B（1B 参数扩散基础模型）在评测环境中的部署与验证，
> 作为从 ACT 行为克隆向具身 VLA 演进的**技术储备**。
> 上游：XPolicyLab/policy/RDT_1B（Apache-2.0），本目录仅含我们的部署说明与验证记录。

## 模型简介

**RDT-1B（Robotic Diffusion Transformer，10 亿参数）** 是面向双臂灵巧操作的**扩散基础模型**：
先在海量双臂数据上预训练通用动作生成器，再按任务微调（「预训练 + 微调」范式）。

```
语言指令 ──► T5-XXL 文本编码器 ──┐
视觉观测 ──► SigLIP 视觉编码器 ──┼─► RDT 扩散 Transformer ──► 动作序列
关节状态 ─► 状态编码器 ─────────┘   （DiT 结构，1B 参数）
```

- **语言**：T5-XXL（42G 权重），开放词汇指令
- **视觉**：SigLIP 图文对齐编码器
- **动作**：扩散模型（DiT）逐步去噪生成动作序列
- **双臂**：D-STEP 统一动作空间，左右臂解耦编码

## 权重清单（48G）

| 组件 | 文件 | 大小 | 说明 |
|---|---|---|---|
| T5-XXL 语言编码器 | `t5-v1_1-xxl/pytorch_model.bin` | 42G | ⚠️ 只有 `pytorch_model.bin`，**无** `model.safetensors` |
| SigLIP 视觉编码器 | `siglip/model.safetensors` | ~2G | |
| RDT-1B 扩散主干 | `rdt-1b/pytorch_model.bin` | ~4G | |

## 环境（conda `rdt_1b`）

- Python 3.10 / PyTorch 2.1 / CUDA 12
- flash-attn 2.7.2（编译时需补 `pkg_resources`/`psutil`）
- numpy 1.26.4（兼容性降级）

## 验证记录（2026-08）

以下模块导入全部通过：
```python
from train.train import train          # 训练入口
from utils.model import T5Embedder      # 语言编码器
from utils.model import SiglipVisionTower  # 视觉编码器
import flash_attn                       # 注意力加速
```

## 训练入口

```
rdt/ 目录内：
  deepspeed main.py        # 训练主入口（finetune.sh 封装）
  bash finetune.sh         # 微调
  bash inference.sh        # 推理
数据先经 process_data.sh 处理为 RDT 格式
```

## 演进计划

1. **参数高效微调**：LoRA + T5 显存卸载，使 1B 扩散模型在受限显存（22G）下可微调；
2. **数据对齐**：将 12 任务数据按 RDT 格式组织，与 ACT 共享预处理管线；
3. **评测对比**：在「泛化维度」上与 ACT 对比，验证 VLA 路线的提升。

## 部署脚本

完整部署脚本见上游 `XPolicyLab/policy/RDT_1B/`：
`install.sh` / `process_data.sh` / `train.sh` / `deploy.py` / `setup_eval_*`。

## 愿景：自然语言控制的双臂机器人

RDT-1B 的长期目标是构建**可用自然语言指令驱动的双臂机器人**（"语音唤醒 → 语言指令 → 双臂操作"的完整交互）。

**数据基础已就绪**：本项目 12 个双臂任务的演示数据**每条 episode 都带语言指令**（hdf5 `instruction` 字段），例如：

> `"Stack the three bowls together."`

这意味着 RDT-1B 可以在天然指令标注的数据上做**语言条件化**微调，无需额外标注。

### 路线图（Phase 1 → 2 → 3）

```
语音层（Phase 2）           VLA 层（Phase 1）                执行层
"小爱同学，把碗叠起来"         RDT-1B 微调                    双臂机器人
  │ 唤醒词检测 + 中文ASR         │ 12任务指令数据 + T5编码        │ arx_x5
  └──────► 文字指令 ──────►  T5编码 + 扩散生成 ──────►     动作输出
```

| 阶段 | 内容 | 目标 | 时间窗 |
|---|---|---|---|
| **Phase 1** | RDT-1B 微调（语言指令→动作） | 文字指令即可驱动 12 任务双臂操作 | 决赛调试窗口（8.25–9.20） |
| **Phase 2** | 语音交互层：唤醒词检测 + 中文 ASR | "语音指令 → 文字指令"（小爱同学式唤醒） | 决赛后迭代 |
| **Phase 3** | 真机部署（arx_x5 实体机器人） | 从仿真到真实双臂操作 | 长期 |

**当前进度**：
- ✅ RDT-1B 部署与权重验证（T5/SigLIP/扩散主干导入通过）
- ✅ 12 任务指令数据确认（含 `instruction` 字段）
- ⏳ 数据转换 + 微调（Phase 1，资源就绪后启动）

**验收标准**：Phase 1 微调版在 12 任务（含泛化维度）成功率 ≥ ACT 基线，方替换 ACT 作为主力策略（详见 `docs/02_技术方案.md` §8.1 两阶段策略）。
