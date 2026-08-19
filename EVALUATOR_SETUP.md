# 评测方部署说明：ACT 双臂策略（GOAI 2026 · 赛题一）

本仓库包含百川汇海团队的双臂 ACT 策略代码、运行配置和 12 任务检查点。模型归档通过 **ModelScope（魔塔）** 交付，由组委会在官方评测环境中加载并评测；不要求参赛方提供公网服务器或网络隧道。

## 交付物分工

| 内容 | 交付渠道 | 说明 |
| --- | --- | --- |
| 策略代码、配置、部署脚本、项目材料 | GitHub：`https://github.com/msd01580158/goai2026-act-bimanual` | 直接克隆仓库 |
| 12 任务检查点 | ModelScope：`https://modelscope.cn/models/msd0158/goai2026-act-bimanual` | 下载 `checkpoints/<task>/` 各目录 |
| 原始训练数据、Isaac Assets、训练日志 | 不交付 | 不属于赛题一代码/权重交付内容 |

检查点每任务含 `policy_last.ckpt`（验证 loss 最优）+ `dataset_stats.pkl`（归一化统计）。SHA-256 见仓库 `MODEL_SHA256SUMS`。

## 目录结构

```text
<repository_root>/
  source/RoboDojo/XPolicyLab/policy/ACT/    # 策略适配层（多任务 + batch 推理）
  config/act_goai.json                      # 模型身份配置
  act/                                     # 训练/部署脚本
  docs/                                    # 技术文档
```

模型归档（从 ModelScope 下载）解压到运行根目录后，检查点位于：

```text
checkpoints/
  <task>/            # 如 stack_bowls/
    policy_last.ckpt
    dataset_stats.pkl
```

策略适配层 `_resolve_task_ckpt_dir` 同时支持 `checkpoints/act-RoboDojo-<task>/<run>/` 与 `checkpoints/<task>/` 两种结构。

## 从下载到官方评测

```bash
cd <repository_root>

# 1. 从 ModelScope 下载 12 任务检查点到 checkpoints/
#    （模型归档：https://modelscope.cn/models/msd0158/goai2026-act-bimanual）
#    逐任务下载 checkpoints/<task>/policy_last.ckpt + dataset_stats.pkl

# 2. 核对 SHA-256（可选，MODEL_SHA256SUMS 提供各文件校验）
# 3. 仓库自检（可选）
python3 -c "import os; p='source/RoboDojo/XPolicyLab/policy/ACT'; assert os.path.isfile(f'{p}/model.py'); print('policy adapter OK')"

# 4. 评测方在官方评测环境中加载以下固定身份
#    policy_name: ACT
#    checkpoint: <repository_root>/checkpoints/<task>/
#    config: config/act_goai.json
#    action_type: joint, action_dim: 14, env_cfg_type: arx_x5
#    camera_names: [cam_head, cam_right_wrist, cam_left_wrist]
```

正式评测请使用组委会提供的官方 evaluator/runner，加载上述 ACT 适配层与检查点，再按官方任务清单运行。任务、布局、seed 和成功判定由官方评测器控制。

## 多任务与 Batch 推理

- **多任务热切换**：发送 `PREPARE_CASE(action_case_id="<task>_case")`，适配层自动加载对应任务检查点；
- **Batch 推理**：`update_obs_batch` / `get_action_batch`，一次提交多环境观测返回多组动作，适配并行评测。

## 评测接入（可选，兼容性自测）

在线端点（供参考，非公网服务要求）：`ws://36.212.51.4:10002`（WebSocket + msgpack，prepare_case 热切换）。
