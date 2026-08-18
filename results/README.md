# 训练结果

本目录记录 12 个任务的训练结果与检查点清单。检查点体积较大（~335MB/个），**不直接入库**，
通过下方两种方式获取。

## 结果总览

| 任务 | 训练批次 | 验证 loss | 最优 epoch | 检查点 | 状态 |
|---|---|---|---|---|---|
| stack_bowls | run1 | _待训练完成_ | _待填_ | _待填_ | 🔄 训练中 |
| make_toast | run1 | _待训练完成_ | _待填_ | _待填_ | 🔄 训练中 |
| arrange_largest_number | run1 | — | — | — | ⏳ 排队 |
| fold_clothes | run1 | — | — | — | ⏳ 排队 |
| hang_mugs | run1 | — | — | — | ⏳ 排队 |
| pack_objects_into_box | run1 | — | — | — | ⏳ 排队 |
| pour_liquid_into_cup | run1 | — | — | — | ⏳ 排队 |
| push_T | run1 | — | — | — | ⏳ 排队 |
| sort_nesting_dolls_by_size | run1 | — | — | — | ⏳ 排队 |
| stack_blocks | run1 | — | — | — | ⏳ 排队 |
| store_laptop_and_headphones | run1 | — | — | — | ⏳ 排队 |
| sweep_blocks | run1 | — | — | — | ⏳ 排队 |

## 检查点获取方式

### 方式 1：在线评测端点（推荐）

训练完成后的策略检查点已部署到云端评测服务器，通过评测端点实时生效：

```
ws://36.212.51.4:10002
```

发送 `PREPARE_CASE(action_case_id="<task>_case")` 即加载对应任务的检查点。

### 方式 2：检查点文件

`policy_best.ckpt`（验证 loss 最优）+ `dataset_stats.pkl` 可打包为
**GitHub Releases** 附件（~335MB/任务）。训练完成后随结果一起发布。

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
