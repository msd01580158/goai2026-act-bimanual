# 双臂灵巧操作智能体（ACT） · GOAI 2026 具身未来赛道

> **Team 百川汇海** · GOAI 2026 · Embodied Future · Track 1: 通用双臂协作操作能力测试
>
> English summary: A bimanual manipulation agent based on **ACT (Action Chunking with
> Transformers)** covering **12 bimanual tasks** on the RoboDojo benchmark, with an
> end-to-end training → cloud deployment pipeline and **multi-task hot-switching**
> (`prepare_case`) on a single evaluation endpoint.

---

## 项目简介

本项目构建了一套覆盖 **12 个双臂操作任务**的灵巧操作智能体，采用 **ACT**（动作分块 Transformer）策略，
在 RoboDojo 仿真基准上完成"感知 → 决策 → 执行"的完整闭环，并实现了**多任务单服务热切换**的云端评测部署。

- **机器人**：arx_x5 双臂机器人，14 维关节动作（左右臂 6+6 关节 + 2 夹爪）
- **观测**：三目视觉（头 + 双腕相机，640×480）+ 14 维关节状态
- **策略**：ACT，ResNet18 backbone，chunk_size=50，时序聚合
- **覆盖任务**：arrange_largest_number / fold_clothes / hang_mugs / make_toast /
  pack_objects_into_box / pour_liquid_into_cup / push_T / sort_nesting_dolls_by_size /
  stack_blocks / stack_bowls / store_laptop_and_headphones / sweep_blocks

## 在线评测端点（Demo）

```
ws://36.212.51.4:10002
```

评测端发送 `PREPARE_CASE(action_case_id="<task>_case")` 即可热切换策略，随后以
`观测 → 动作` 闭环推进评测。详见 [docs/Demo说明.md](docs/Demo说明.md)。

## 系统架构

```
评测系统
   │ ws://36.212.51.4:10002 (WebSocket + msgpack)
   ▼
[云端评测服务器 2×A100-40GB]
   │ tcp 转发 10002 → 6000
   ▼
[ACT Policy Server]
   ├─ model.py          ← prepare_case 多任务热切换（本仓库核心贡献）
   └─ checkpoints/act-RoboDojo-<task>/<run>/policy_last.ckpt + dataset_stats.pkl
```

训练在本机双卡进行（RTX 2080 Ti 22G + RTX 4000 Ada 11.5G），检查点上传服务器即生效。

## 本仓库内容（贡献代码）

| 路径 | 说明 | 相对上游的改动 |
|---|---|---|
| `act/model_multi_task.py` | 多任务策略模型 | 新增 `prepare_case` 钩子，按 `action_case_id` 热切换检查点 |
| `act/imitate_episodes.py` | ACT 训练入口 | 保存验证集 loss 最优检查点 `policy_best.ckpt` |
| `act/process_data.py` | RoboDojo 数据 → ACT 格式 | 图像逐帧 gzip 分块存储（磁盘占用降约 2/3） |
| `act/train.sh` / `process_data.sh` | 训练 / 数据处理封装 | — |
| `act/TASK_CONFIGS.json.sample` | 任务配置样例 | — |
| `docs/` | 项目介绍 / 技术方案 / Demo 说明 / 部署与依赖 | — |

> 本仓库基于开源 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 与
> [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) 构建，**不包含**两者的完整源码、
> 大赛数据集（155G）与训练检查点；请从上游获取。署名见 [NOTICE](NOTICE)。

## 快速开始

### 依赖
- PyTorch ≥ 2.1（推荐 2.4+），torchvision，NumPy 1.26.4，h5py，OpenCV，einops
- XPolicyLab（editable 安装）、RoboDojo（数据与评测）

### 训练
```bash
cd <RoboDojo>/XPolicyLab/policy/ACT
# 1) 数据转换（自动生成 TASK_CONFIGS.json）
bash process_data.sh RoboDojo <task> arx_x5 joint
# 2) 训练（输出 policy_last.ckpt / policy_best.ckpt / dataset_stats.pkl）
export ACT_ACTION_DIM=14 CUDA_VISIBLE_DEVICES=1
python imitate_episodes.py --bench_name RoboDojo --task_name <task> \
  --ckpt_setting RoboDojo-<task>-arx_x5-joint \
  --ckpt_dir checkpoints/RoboDojo-<task>-arx_x5-joint-run1 \
  --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
  --dim_feedforward 3200 --batch_size 16 --num_epochs 2000 --lr 1e-5 \
  --save_freq 500 --seed 0
```

### 部署
将 `act/model_multi_task.py` 放入 `XPolicyLab/policy/ACT/model.py`，检查点放入
`checkpoints/act-RoboDojo-<task>/<run>/`（含 `policy_last.ckpt` + `dataset_stats.pkl`），
评测端即可通过 `prepare_case` 热切换。详细步骤见 [docs/部署与依赖说明.md](docs/部署与依赖说明.md)。

## 许可证
Apache-2.0，署名见 [NOTICE](NOTICE)。上游项目许可证见各仓库。

## 致谢
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) — 仿真基准与评测框架
- [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) — 策略集成与部署框架
- [ACT](https://tonyzhaozh.github.io/ACT/) — 动作分块 Transformer 策略
