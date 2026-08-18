# checkpoints/

训练检查点（`policy_last.ckpt` + `dataset_stats.pkl`）体积较大（约 335MB/个 × 12 任务），
**不在本仓库分发**。它们部署于云端评测服务器：

```
~/RoboDojo/XPolicyLab/policy/ACT/checkpoints/act-RoboDojo-<task>/<run>/
    ├── policy_last.ckpt
    ├── policy_best.ckpt        # 验证集 loss 最优
    └── dataset_stats.pkl
```

如需复现评测，请按 [docs/04_部署与依赖说明.md](../docs/04_部署与依赖说明.md) 的步骤自行训练，
或联系维护者获取检查点。
