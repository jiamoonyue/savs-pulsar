"""构建区域感知 SAVs 数据：读 pulsar_clip crops_index.csv，4 面板，随机 few-shot 划分。

与 build_pulsar_data.py 的区别：样本是"4 张面板子图"，而非整图。
面板：freq_phase / time_phase / dm_curve / profile（忽略 text_info 文本区域）。

用法:
  python scripts/build_region_data.py --k 25 --seed 0 --rfi_query 2000
"""
import argparse
import csv
import json
import os
import random

CSV_PATH = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops_index.csv"
LINUX_CROPS = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops"
PANELS = ["freq_phase", "time_phase", "dm_curve", "profile"]  # 4 物理面板，忽略 text_info


def to_linux_path(win_path):
    """D:\\ApJS实验\\pulsar_clip\\artifacts\\crops\\u000000\\profile.png
    -> /SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops/u000000/profile.png"""
    return win_path.replace("\\", "/").replace("D:/", "/SAVS_DATA_ROOT/")


def load_samples():
    """读 crops_index.csv，返回 [{uid, label, panels:{freq_phase,time_phase,dm_curve,profile}}, ...]"""
    samples = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            panels = {}
            ok = True
            for p in PANELS:
                linux = to_linux_path(row[p])
                if not os.path.exists(linux):
                    ok = False
                    break
                panels[p] = linux
            if not ok:
                continue
            samples.append({"uid": row["uid"], "label": row["label"], "panels": panels})
    return samples


def split(samples, k_per_class=25, seed=0, rfi_query=2000):
    """随机抽每类 k 张做支持集（平衡）；查询集 = 剩余全部 pulsar + rfi 子采样 rfi_query 张（轻度不平衡）。"""
    rng = random.Random(seed)
    by_label = {"pulsar": [], "rfi": []}
    for s in samples:
        by_label[s["label"]].append(s)

    support = []
    for label in ("pulsar", "rfi"):
        pool = by_label[label][:]
        rng.shuffle(pool)
        support.extend(pool[:k_per_class])
        by_label[label] = pool[k_per_class:]

    pulsar_query = by_label["pulsar"]           # 全部剩余 pulsar（~1135）
    rfi_pool = by_label["rfi"][:]
    rng.shuffle(rfi_pool)
    rfi_query_sel = rfi_pool[:rfi_query]         # rfi 子采样到 rfi_query
    query = pulsar_query + rfi_query_sel
    return support, query


def write_jsonl(path, samples):
    with open(path, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25, help="每类支持集数量")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--rfi_query", type=int, default=2000, help="查询集 rfi 子采样数量")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "region"))
    args = ap.parse_args()

    samples = load_samples()
    support, query = split(samples, k_per_class=args.k, seed=args.seed, rfi_query=args.rfi_query)
    os.makedirs(args.out, exist_ok=True)
    all_path = os.path.join(args.out, "all.jsonl")
    sup_path = os.path.join(args.out, f"support_k{args.k}_s{args.seed}.jsonl")
    qry_path = os.path.join(args.out, f"query_k{args.k}_s{args.seed}.jsonl")
    write_jsonl(all_path, samples)
    write_jsonl(sup_path, support)
    write_jsonl(qry_path, query)
    print(f"all={len(samples)} (pulsar {sum(1 for s in samples if s['label']=='pulsar')} / rfi {sum(1 for s in samples if s['label']=='rfi')})")
    print(f"support={len(support)} query={len(query)} (pulsar {sum(1 for s in query if s['label']=='pulsar')} / rfi {sum(1 for s in query if s['label']=='rfi')})")
    print("样例面板:", json.dumps(support[0]["panels"] if support else {}, indent=2)[:200])
