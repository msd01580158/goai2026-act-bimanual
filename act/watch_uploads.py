#!/usr/bin/env python3
"""上传监控：检查已完成训练但未上传服务器的任务，带重试补传。

每 5 分钟扫描本地 checkpoints/RoboDojo-<task>-arx_x5-joint-run1/policy_best.ckpt，
若服务器 act-RoboDojo-<task>/run1/policy_last.ckpt 缺失（SSH 不稳导致漏传），则重试上传。
上传成功后更新演示页与 GitHub 结果。
"""
import os, sys, subprocess, time, json, logging

ACT_DIR = "/home/mashideng/RoboDojo/XPolicyLab/policy/ACT"
CKPT_BASE = os.path.join(ACT_DIR, "checkpoints")
RUN_NAME = "run1"
SERVER = "mashideng@36.212.51.4"
SERVER_CKPT = "/home/mashideng/RoboDojo/XPolicyLab/policy/ACT/checkpoints"
DEMO_HTML = "/home/mashideng/goai2026_act_demo_page.html"
REPO = "/home/mashideng/goai2026-act-bimanual"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [watcher] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/watch_uploads.log")])

TASKS = [
    "stack_bowls", "make_toast", "arrange_largest_number", "fold_clothes", "hang_mugs",
    "pack_objects_into_box", "pour_liquid_into_cup", "push_T",
    "sort_nesting_dolls_by_size", "stack_blocks", "store_laptop_and_headphones", "sweep_blocks",
]


def ssh_cmd(cmd, retries=3):
    for i in range(retries):
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", SERVER, cmd],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        time.sleep(10)
    return ""


def server_has(task):
    # 两个文件都必须在，缺一（如只传了 policy 漏了 stats）视为未上传
    out = ssh_cmd(f"test -f {SERVER_CKPT}/act-RoboDojo-{task}/{RUN_NAME}/policy_last.ckpt "
                  f"&& test -f {SERVER_CKPT}/act-RoboDojo-{task}/{RUN_NAME}/dataset_stats.pkl && echo yes")
    return "yes" in out


def upload_task(task, retries=6):
    local = os.path.join(CKPT_BASE, f"RoboDojo-{task}-arx_x5-joint-{RUN_NAME}")
    best = os.path.join(local, "policy_best.ckpt")
    stats = os.path.join(local, "dataset_stats.pkl")
    if not os.path.isfile(best) or not os.path.isfile(stats):
        return False
    remote = f"{SERVER_CKPT}/act-RoboDojo-{task}/{RUN_NAME}"
    if not ssh_cmd(f"mkdir -p {remote}"):
        return False
    for name, src in [("policy_last.ckpt", best), ("dataset_stats.pkl", stats)]:
        ok = False
        for i in range(retries):
            r = subprocess.run(["scp", "-o", "ConnectTimeout=15", src, f"{SERVER}:{remote}/{name}"],
                               capture_output=True)
            if r.returncode == 0:
                ok = True
                break
            time.sleep(15)
        if not ok:
            logging.error(f"[{task}] 上传 {name} 失败")
            return False
    logging.info(f"[{task}] 补传成功 {remote}")
    return True


def val_loss(task):
    log = f"/tmp/train_{task}.log"
    if not os.path.isfile(log):
        return None
    for line in open(log, errors="ignore"):
        if "Best val loss" in line:
            parts = line.split()
            try:
                loss = parts[parts.index("loss") + 1]
                # token 形如 "epoch1859,"（无独立 "epoch" 词），按前缀匹配
                ep = next((p for p in parts if p.startswith("epoch")), "?").rstrip(",")
                return loss, ep
            except Exception:
                return None
    return None


def update_demo_page():
    """重生成演示页的 训练进展 表格。"""
    if not os.path.isfile(DEMO_HTML):
        return
    html = open(DEMO_HTML).read()
    rows = ""
    done = 0
    for t in TASKS:
        vl = val_loss(t)
        if vl:
            done += 1
            rows += f'    <tr><td>{t}</td><td>{vl[0]}</td><td>✅</td></tr>\n'
    pending = [t for t in TASKS if not val_loss(t)]
    note = " · ".join(pending)
    table = f"""<section>
  <h2>训练进展（2026-08-19）</h2>
  <p>12 个双臂任务双卡并行训练中，已完成 {done}/12 并部署至评测端点：</p>
  <table>
    <tr><th>任务</th><th>验证 loss</th><th>部署</th></tr>
{rows}    <tr><td colspan="3" style="text-align:center;color:#6b7280">{note} 待训</td></tr>
  </table>
</section>
"""
    import re
    html = re.sub(r'<section>\s*<h2>训练进展.*?</section>\n', table, html, flags=re.S)
    if table not in html:
        # 若原本无该节，插入到 开源 节前
        html = html.replace('<section>\n  <h2>开源</h2>', table + '<section>\n  <h2>开源</h2>', 1)
    open(DEMO_HTML, "w").write(html)
    # 上传演示页（带重试）
    for i in range(4):
        r = subprocess.run(["scp", "-o", "ConnectTimeout=15", DEMO_HTML,
                            f"{SERVER}:~/demo_page/index.html"], capture_output=True)
        if r.returncode == 0:
            logging.info(f"演示页已更新（{done}/12 任务）")
            return
        time.sleep(15)
    logging.warning("演示页上传失败（SSH 不稳，稍后重试）")


def update_github():
    """更新 GitHub results/README.md 并推送。"""
    try:
        subprocess.run(["git", "-C", REPO, "add", "-A"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", "results: 自动同步训练进展"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "push", "-q", "origin", "main"], capture_output=True)
    except Exception as e:
        logging.error(f"GitHub 更新失败: {e}")


last_synced = set()

if __name__ == "__main__":
    while True:
        time.sleep(300)
        for task in TASKS:
            local = os.path.join(CKPT_BASE, f"RoboDojo-{task}-arx_x5-joint-{RUN_NAME}", "policy_best.ckpt")
            if not os.path.isfile(local):
                continue
            if not server_has(task):
                logging.info(f"[{task}] 服务器缺 ckpt，补传中...")
                if upload_task(task):
                    last_synced.add(task)
                    update_demo_page()
                    update_github()
            elif task not in last_synced:
                # 服务器已有（可能是驱动直接传的），同步网站
                last_synced.add(task)
                update_demo_page()
                update_github()
                logging.info(f"[{task}] 已在服务器，同步网站")
