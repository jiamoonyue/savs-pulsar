"""统一 few-shot 数据划分 —— 整图与四图共享同一批 uid，保证支持/查询集逐样本对齐。

为什么需要（历史问题）:
  * 旧 build_pulsar_data.py（整图）与 build_region_gpps_data.py（四图）各自独立随机
    划分，导致同一 (k, seed) 下两法的支持集/查询集不是同一批图——"整图 vs 四图"
    的指标差被划分差异污染（实测 k25/s0 支持集仅 3/50 重叠，查询集 ~44% 重叠）。
  * region 版把正类支持集写死（pulsar 走固定 query_seed 洗牌），seed 只影响负类，
    多 seed 的 ±std 低估真实方差。

本模块是唯一划分入口（split 对任意 {uid,label} 样本列表通用）。
同一 (k, seed) 下，整图与四图派生出完全相同的支持集与查询集 uid（重叠 100%）。

划分协议:
  * 查询集固定（不随 seed / k 变）:
      - 每类保留 reserve 张作支持池，其余可进查询集；
      - rfi 查询抽 rfi_query 张（上限 = 池 - reserve）；pulsar 查询取"池 - reserve"。
  * 支持集: 每类 k 张，从"查询集之外"的池子按 seed 重抽（正负两类都重抽）。
"""
import csv
import os
import random
import re

MANIFEST = "/SAVS_DATA_ROOT/ApJS实验/FAST_split/manifest_full.csv"
CROPS_INDEX = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops_index.csv"
PIC_ROOT = "/SAVS_DATA_ROOT/CCF-A/datasets/FAST/PICS-ResNet_data_png"
PANELS = ["freq_phase", "time_phase", "dm_curve", "profile"]


def to_linux_path(win_path):
    return win_path.replace("\\", "/").replace("D:/", "/SAVS_DATA_ROOT/")


def is_gpps(file, target=""):
    """GPPS 筛选：剔除 M31 星系巡天与已知源(J 坐标)。"""
    b = os.path.basename(file)
    if "M31" in b or "M31" in target:
        return False
    if re.search(r"_J\d{4}[+-]\d", b):
        return False
    return True


def load_pool(verify_files=True):
    """GPPS 样本池（仅保留整图与 4 面板都存在者），按 uid 升序。

    每个元素: {"uid", "label", "whole", "panels": {panel: path}}
    """
    with open(MANIFEST, encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)]
    rows.sort(key=lambda r: r["file"])
    uid2row = {f"u{i:06d}": r for i, r in enumerate(rows)}

    crops = {}
    with open(CROPS_INDEX, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            crops[r["uid"]] = r

    pool = []
    for uid, row in uid2row.items():
        if not is_gpps(row["file"], row.get("target", "")):
            continue
        c = crops.get(uid)
        if c is None:
            continue
        whole = PIC_ROOT + "/" + row["file"].replace("\\", "/")
        panels = {p: to_linux_path(c[p]) for p in PANELS}
        if verify_files:
            if not os.path.exists(whole):
                continue
            if not all(os.path.exists(v) for v in panels.values()):
                continue
        pool.append({"uid": uid, "label": row["label"], "whole": whole, "panels": panels})
    return pool


def split(samples, k_per_class=25, seed=0, rfi_query=2000, query_seed=42, reserve=100):
    """通用 few-shot 划分 → (support, query)，均为样本 dict 列表。

    适用于任意 {"uid","label", ...} 样本列表（A=GPPS / B=ACCEL / Parkes 共用）。
    查询集固定（query_seed）；支持集按 seed 从查询集之外的池子重抽（正负两类都抽）。
    reserve: 每类为支持池保留的最少样本数（须 > k 才能让支持集随 seed 变）。
    """
    k_per_class = int(k_per_class)
    keep = max(int(reserve), k_per_class)

    by = {}
    for s in samples:
        by.setdefault(s["label"], []).append(s)

    # --- 固定查询集 ---
    qrng = random.Random(query_seed)
    query, q_uid = [], set()
    for label in sorted(by):
        p = by[label][:]
        qrng.shuffle(p)
        room = max(len(p) - keep, 0)
        want = int(rfi_query) if label == "rfi" else room
        sel = p[:max(min(want, room), 0)]
        query += sel
        q_uid |= {s["uid"] for s in sel}

    # --- 支持集（随 seed 重抽，正负两类都抽） ---
    rng = random.Random(seed)
    support = []
    for label in sorted(by):
        cand = [s for s in by[label] if s["uid"] not in q_uid]
        rng.shuffle(cand)
        support += cand[:k_per_class]
    return support, query
