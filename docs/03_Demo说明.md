# Demo 说明

## 赛道背景
本 Demo 对应**具身未来（Embodied Future）· 赛题一：通用双臂协作操作能力测试**（X-Eval 仿真与真机一体化评测平台），围绕双臂协同操作展示"感知 → 决策 → 执行"的完整闭环。

## 评测接入方式

评测系统通过**公网 WebSocket 端点**接入我们的双臂策略服务：

```
ws://36.212.51.4:10002
```

- **传输协议**：WebSocket + msgpack 二进制帧。
- **消息类型**（`MessageType`）：`PREPARE_CASE`、观测（`OBS`）、动作（`ACTION`）等。
- **机器人/动作类型**：arx_x5 双臂机器人，关节（`joint`）动作，14 维。

## 使用流程

### 1. 选择任务（多任务热切换）

发送 `PREPARE_CASE` 消息，携带：

```
action_case_id = "<任务名>_case"
```

示例：

| 任务 | action_case_id |
|---|---|
| stack_bowls | `stack_bowls_case` |
| push_T | `push_T_case` |
| arrange_largest_number | `arrange_largest_number_case` |
| …（共 12 个任务） | `<任务名>_case` |

服务端据此自动加载对应任务的策略检查点，无需重启。

### 2. 发送观测，接收动作

- **观测**（`OBS`）：包含三目视觉（`cam_head`、`cam_left_wrist`、`cam_right_wrist`，640×480 彩色图）与机器人关节状态。
- **动作**（`ACTION`）：返回 14 维关节动作，由策略的时序聚合模块按动作块平滑输出。

### 3. 评测闭环

`观测 → 策略推理 → 动作 → 仿真环境执行 → 新观测` 循环推进，直至任务完成/失败。

## 已部署任务清单（12/12）

arrange_largest_number、fold_clothes、hang_mugs、make_toast、pack_objects_into_box、pour_liquid_into_cup、push_T、sort_nesting_dolls_by_size、stack_blocks、stack_bowls、store_laptop_and_headphones、sweep_blocks

## 服务可用性

- **服务状态**：评测服务 7×24 运行于云端（36.212.51.4，2×A100-40GB），端口 `10002`（公网）→ `6000`（策略服务）TCP 转发。
- **多任务验证**：`prepare_case` 钩子已通过端到端验证——本地训练检查点上传后可被服务实时加载推理，兼容跨 torch 版本（本机 torch 2.7 训练 → 服务器 torch 2.4 加载）。

## 演示/展示要点

1. **多任务统一端点**：单个 `ws://` 地址即可依次评测 12 个任务，演示多任务热切换的便捷性；
2. **端到端管线**：从演示数据到可评测策略的完整自动化（数据预处理 → 训练 → 检查点 → 云端部署）；
3. **策略迭代**：训练产出的最优检查点可随时上传替换线上策略，支持持续优化。

## 联系与复现

- 代码仓库：
  - RoboDojo：`https://github.com/RoboDojo-Benchmark/RoboDojo`
  - XPolicyLab：`https://github.com/XPolicyLab/XPolicyLab`
- 评测服务与本方案文档由本项目团队维护。

---

## 开源仓库

本项目贡献代码与文档开源在：**https://github.com/msd01580158/goai2026-act-bimanual**（Apache-2.0）

