#!/usr/bin/env python3
"""GOAI 2026 ACT 批量训练驱动：双卡并行，转换→训练→上传→清理。

用法：
  EPOCHS=2000 SAVE_FREQ=500 python3 train_act_batch.py

逻辑：
  - 12 个 GOAI 任务队列，2 个工作线程分别绑定 GPU0(batch8)/GPU1(batch16)
  - 每任务：数据未转换则转换 → 训练 → 上传 policy_best.ckpt 到服务器 → 删除本地转换数据
  - TASK_CONFIGS.json 写操作串行化（转换加锁）
  - 可断点：服务器上已存在 run1 检查点的任务自动跳过
"""
import os, sys, time, json, shutil, subprocess, threading, logging

ACT_DIR = "/home/mashideng/RoboDojo/XPolicyLab/policy/ACT"
PY = "/home/mashideng/anaconda3/envs/lyra2/bin/python"
SERVER = "mashideng@36.212.51.4"
SERVER_CKPT = "/home/mashideng/RoboDojo/XPolicyLab/policy/ACT/checkpoints"
RUN_NAME = "run1"
EPOCHS = int(os.environ.get("EPOCHS", "2000"))
SAVE_FREQ = int(os.environ.get("SAVE_FREQ", "500"))
CONFIG_LOCK = threading.Lock()   # 仅保护 TASK_CONFIGS.json 写操作
PRECONVERT_DISK_GB = int(os.environ.get("PRECONVERT_DISK_GB", "48"))  # 预转换磁盘保护阈值（单个转换约需 41G）
converting = set()               # 正在被转换的任务（防双转换）

TASKS = [
    # 已转换的任务优先（双卡立即并行训练）
    "stack_bowls", "make_toast",
    "arrange_largest_number", "fold_clothes", "hang_mugs",
    "pack_objects_into_box", "pour_liquid_into_cup", "push_T",
    "sort_nesting_dolls_by_size", "stack_blocks",
    "store_laptop_and_headphones", "sweep_blocks",
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/train_act_batch.log")],
)


def converted_ok(task):
    p = os.path.join(ACT_DIR, "processed_data", "RoboDojo", task, "arx_x5-joint", "episode_99.hdf5")
    return os.path.isfile(p)


def server_ckpt_ok(task, retries=4):
    # 远程检查点已存在则跳过。SSH 不稳时重试，避免瞬时失败误判为"不在服务器"而重训已完成任务。
    for i in range(retries):
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", SERVER,
             f"test -f {SERVER_CKPT}/act-RoboDojo-{task}/{RUN_NAME}/policy_last.ckpt && echo yes"],
            capture_output=True, text=True)
        if r.returncode == 0:
            return "yes" in r.stdout
        time.sleep(10)
    logging.warning(f"[{task}] SSH 多次失败，无法确认服务器状态，按未上传处理（宁肯重训也不丢任务）")
    return False


def convert_task(task):
    # 并行转换（不同任务互不冲突；TASK_CONFIGS 由 CONFIG_LOCK + ensure 自愈）
    if converted_ok(task):
        logging.info(f"[{task}] 转换已完成，跳过")
        return True
    logging.info(f"[{task}] 开始转换数据")
    r = subprocess.run(
        [PY, "detr/process_data.py", "RoboDojo", task, "arx_x5", "joint"],
        cwd=ACT_DIR, capture_output=True, text=True)
    if r.returncode != 0 or not converted_ok(task):
        logging.error(f"[{task}] 转换失败:\n{r.stderr[-2000:]}")
        return False
    logging.info(f"[{task}] 转换完成")
    return True


def ensure_task_config(task):
    """确保 TASK_CONFIGS.json 有当前任务条目（并行转换可能丢条目，自愈补齐）。"""
    cfg_path = os.path.join(ACT_DIR, "TASK_CONFIGS.json")
    with CONFIG_LOCK:
        cfg = {}
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        key = f"RoboDojo-{task}-arx_x5-joint"
        if key in cfg:
            return True
        data_dir = os.path.join(ACT_DIR, "processed_data", "RoboDojo", task, "arx_x5-joint")
        n = len([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".hdf5")]) if os.path.isdir(data_dir) else 0
        if n == 0:
            return False
        cfg[key] = {
            "dataset_dir": os.path.join("processed_data", "RoboDojo", task, "arx_x5-joint"),
            "num_episodes": n,
            "episode_len": 5000,
            "camera_names": ["cam_head", "cam_right_wrist", "cam_left_wrist"],
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4)
        logging.info(f"[{task}] TASK_CONFIGS 补齐条目 (episodes={n})")
        return True


def train_task(task, gpu, batch):
    ckpt_dir = f"checkpoints/RoboDojo-{task}-arx_x5-joint-{RUN_NAME}"
    env = dict(os.environ, ACT_ACTION_DIM="14", CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, "imitate_episodes.py",
           "--bench_name", "RoboDojo", "--task_name", task,
           "--ckpt_setting", f"RoboDojo-{task}-arx_x5-joint",
           "--ckpt_dir", ckpt_dir,
           "--policy_class", "ACT", "--kl_weight", "10", "--chunk_size", "50",
           "--hidden_dim", "512", "--dim_feedforward", "3200",
           "--batch_size", str(batch), "--num_epochs", str(EPOCHS),
           "--lr", "1e-5", "--save_freq", str(SAVE_FREQ), "--seed", "0"]
    logging.info(f"[{task}] GPU{gpu} 训练开始 (batch={batch}, epochs={EPOCHS})")
    logf = open(f"/tmp/train_{task}.log", "w")
    r = subprocess.run(cmd, cwd=ACT_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    return r.returncode == 0 and os.path.isfile(os.path.join(ACT_DIR, ckpt_dir, "policy_best.ckpt"))


def upload_task(task):
    local = os.path.join(ACT_DIR, "checkpoints", f"RoboDojo-{task}-arx_x5-joint-{RUN_NAME}")
    best = os.path.join(local, "policy_best.ckpt")
    stats = os.path.join(local, "dataset_stats.pkl")
    remote_dir = f"{SERVER_CKPT}/act-RoboDojo-{task}/{RUN_NAME}"
    steps = [
        ["ssh", "-o", "ConnectTimeout=8", SERVER, f"mkdir -p {remote_dir}"],
        ["scp", "-q", best, f"{SERVER}:{remote_dir}/policy_last.ckpt"],
        ["scp", "-q", stats, f"{SERVER}:{remote_dir}/dataset_stats.pkl"],
    ]
    for s in steps:
        if subprocess.run(s, capture_output=True).returncode != 0:
            logging.error(f"[{task}] 上传步骤失败: {s}")
            return False
    logging.info(f"[{task}] 已上传 policy_last.ckpt + dataset_stats.pkl 到 {remote_dir}")
    return True


def cleanup_task(task):
    # 删除本地转换数据与训练中间检查点（保留 server 上的成果）
    subprocess.run(["rm", "-rf", os.path.join(ACT_DIR, "processed_data", "RoboDojo", task)], capture_output=True)
    subprocess.run(["rm", "-rf", os.path.join(ACT_DIR, "checkpoints", f"RoboDojo-{task}-arx_x5-joint-{RUN_NAME}")], capture_output=True)
    logging.info(f"[{task}] 本地转换数据已清理")


def preconvert_worker(todo, done):
    """后台预转换线程：趁当前任务训练时，把下一个未转换的任务数据先转好，交接时 GPU 零空闲。
    磁盘保护：可用空间低于阈值则跳过本轮；与 worker 的 convert_task 共用 CONV_LOCK 串行。"""
    while True:
        time.sleep(90)
        with done_lock:
            if converting:
                # 有 worker 在转换（或上一次 preconvert 未结束），跳过避免磁盘堆叠
                continue
            task = next((t for t in todo
                         if t not in done and not converted_ok(t)
                         and not server_ckpt_ok(t)), None)
            if task is None:
                continue
            free_gb = shutil.disk_usage(ACT_DIR).free / 2**30
            # 预转换需要容纳"本任务转换(约41G) + 已有训练中任务(约41G)"，阈值按 2×转换量保护
            if free_gb < 2 * PRECONVERT_DISK_GB:
                logging.info(f"[preconvert] 磁盘不足({free_gb:.0f}G < {2*PRECONVERT_DISK_GB}G)，本轮跳过")
                continue
            converting.add(task)
        logging.info(f"[preconvert] 预转换 {task} (磁盘 {free_gb:.0f}G)")
        r = subprocess.run([PY, "detr/process_data.py", "RoboDojo", task, "arx_x5", "joint"],
                           cwd=ACT_DIR, capture_output=True)
        with done_lock:
            converting.discard(task)
        if r.returncode == 0 and converted_ok(task):
            logging.info(f"[preconvert] {task} 预转换完成，随时可训练")
        else:
            logging.error(f"[preconvert] {task} 预转换失败")


def worker(gpu, batch, todo, done):
    while True:
        with done_lock:
            task = next((t for t in todo if t not in done and t not in converting), None)
            if task is None:
                return
            done.add(task)          # 占坑，防止两个 worker 取同一任务
            converting.add(task)    # 认领转换，防 preconvert 重复转换
        try:
            if server_ckpt_ok(task):
                logging.info(f"[{task}] 服务器已有 {RUN_NAME} 检查点，跳过")
            elif not convert_task(task):
                logging.error(f"[{task}] 转换失败")
            else:
                # 转换完成即释放 converting 认领——训练阶段 preconvert 可预转换下一个任务
                with done_lock:
                    converting.discard(task)
                if not ensure_task_config(task):
                    logging.error(f"[{task}] TASK_CONFIGS 无法补齐")
                elif not train_task(task, gpu, batch):
                    logging.error(f"[{task}] 训练失败，保留数据便于排查")
                elif upload_task(task):
                    cleanup_task(task)
        except Exception as e:
            logging.error(f"[{task}] 异常: {e}")
        finally:
            with done_lock:
                converting.discard(task)


done_lock = threading.Lock()
done = set()
threads = []
for gpu, batch in [(1, 16), (0, 8)]:
    t = threading.Thread(target=worker, args=(gpu, batch, TASKS, done), name=f"GPU{gpu}")
    t.start()
    threads.append(t)
# 后台预转换线程（减少 GPU 交接空转）
pre = threading.Thread(target=preconvert_worker, args=(TASKS, done), name="preconvert", daemon=True)
pre.start()
threads.append(pre)
for t in threads:
    t.join()

logging.info("=== 批量训练全部结束 ===")
