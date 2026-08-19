# 训练结果

12 个双臂任务的训练结果与检查点清单。检查点体积较大（~335MB/个），**不直接入库**，
通过在线评测端点（部署到云端）获取，训练完成后随 GitHub Releases 发布文件。

## 结果总览（更新于 2026-08-19）

| 任务 | 训练批次 | 验证 loss | 最优 epoch | 部署 | 状态 |
|---|---|---|---|---|---|
| stack_bowls | run1 | **0.1575** | 1859 | ✅ | ✅ 已上传 |
| make_toast | run1 | **0.2045** | 1832 | ✅ | ✅ 已上传 |
| arrange_largest_number | run1 | **0.0885** | 1821 | ✅ | ✅ 已上传 |
| fold_clothes | run1 | **0.0525** | 1960 | ✅ | ✅ 已上传 |
| hang_mugs | run1 | **0.1927** | 1340 | ✅ | ✅ 已上传 |
| pack_objects_into_box | run1 | _训练中_ | — | — | 🔄 训练中 |
| pour_liquid_into_cup | run1 | _训练中_ | — | — | 🔄 训练中 |
| push_T | run1 | — | — | — | ⏳ 排队 |
| sort_nesting_dolls_by_size | run1 | — | — | — | ⏳ 排队 |
| stack_blocks | run1 | — | — | — | ⏳ 排队 |
| store_laptop_and_headphones | run1 | — | — | — | ⏳ 排队 |
| sweep_blocks | run1 | — | — | — | ⏳ 排队 |

> 训练方法：ACT，2000 epochs，验证集筛选最优检查点（详见 `docs/02_技术方案.md` §5）。

## 训练过程截图（证据）

- [训练过程.png](screenshots/训练过程.png)
- [训练过程截图.png](screenshots/训练过程截图.png)
- [训练过程截图2.png](screenshots/训练过程截图2.png)

## 检查点获取方式

### 方式 1：在线评测端点（推荐）

训练完成的策略检查点已部署到云端评测服务器，通过评测端点实时生效：

```
ws://36.212.51.4:10002
```

发送 `PREPARE_CASE(action_case_id="<task>_case")` 即加载对应任务的检查点。

### 方式 2：检查点文件（ModelScope）

`policy_best.ckpt`（验证 loss 最优）+ `dataset_stats.pkl`（~335MB/任务）将托管于
**魔塔社区（ModelScope）**：

> 🔗 **ModelScope 检查点仓库：`https://modelscope.cn/models/msd0158/goai2026-act-bimanual`**

全部任务训练完成后统一发布。

## 训练方法（可复现）

每任务训练命令见 `act/` 目录与 `docs/04_部署与依赖说明.md`：

```bash
export ACT_ACTION_DIM=14 CUDA_VISIBLE_DEVICES=1
python imitate_episodes.py --bench_name RoboDojo --task_name <task> \
  --ckpt_setting RoboDojo-<task>-arx_x5-joint \
  --ckpt_dir checkpoints/RoboDojo-<task>-arx_x5-joint-run1 \
  --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
  --dim_feedforward 3200 --batch_size 16 --num_epochs 2000 --lr 1e-5 \
  --save_freq 500 --seed 0
```
