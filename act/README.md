# act/ — 贡献代码说明

本目录包含我们在开源 XPolicyLab / RoboDojo 基础之上的**贡献代码**，均源自
`XPolicyLab/policy/ACT`（Apache-2.0），附带我们的修改。

| 文件 | 上游来源 | 我们的修改 |
|---|---|---|
| `model_multi_task.py` | `XPolicyLab/policy/ACT/model.py` | **新增 `prepare_case` 多任务热切换钩子**：按 `action_case_id`（`<task>_case`）解析并加载 `checkpoints/act-RoboDojo-<task>/` 下含 `policy_last.ckpt` + `dataset_stats.pkl` 的检查点；同任务重复请求仅重置时序状态 |
| `imitate_episodes.py` | `XPolicyLab/policy/ACT/imitate_episodes.py` | **保存验证集 loss 最优检查点 `policy_best.ckpt`**（用于筛选最优 epoch） |
| `process_data.py` | `XPolicyLab/policy/ACT/detr/process_data.py` | **图像逐帧 gzip 分块存储**（`compression="gzip"` + 每帧 chunk），磁盘占用降约 2/3，随机读取仍快 |
| `train.sh` / `process_data.sh` | 上游 | 无修改（训练/转换命令封装） |
| `TASK_CONFIGS.json.sample` | 自动生成 | 样例 |

> 完整模型权重与 `detr/` 实现来自上游 XPolicyLab/ACT，未在此重复分发。

## 修改点快速定位

```bash
# prepare_case（model_multi_task.py）
grep -n "prepare_case\|_resolve_task_ckpt_dir" model_multi_task.py

# 最优检查点保存（imitate_episodes.py）
grep -n "policy_best" imitate_episodes.py

# gzip 压缩（process_data.py）
grep -n "compression\|chunks" process_data.py
```

### `train_act_batch.py` / `watch_uploads.py`（批量训练驱动与上传兜底）

- **`train_act_batch.py`**：12 任务双卡并行批量训练驱动——自动「转换→训练→上传→清理」循环，预转换线程消除 GPU 交接空转，SSH 检查带重试（详见 `docs/06_工程实践与部署经验.md`）。
- **`watch_uploads.py`**：上传兜底 watcher——扫描"本地有检查点但服务器缺失"的任务自动补传（rsync 断点续传 + MD5 校验），上传成功同步演示页与结果清单。
