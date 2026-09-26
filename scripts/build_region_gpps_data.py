"""构建 GPPS 纯化的【四图(4 面板)】数据 —— 与整图共享同一 uid 划分。

统一划分入口见 gpps_data.py。整图与四图同 (k, seed) → 支持/查询集逐样本相同。

用法:
  python scripts/build_region_gpps_data.py --k 25 --seed 0 --rfi_query 2000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpps_data import load_pool, split  # noqa: E402


def to_region(s):
    return {"uid": s["uid"], "label": s["label"], "panels": s["panels"]}


def write_jsonl(path, items):
    with open(path, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


def label_counts(items):
    n_pul = sum(1 for s in items if s["label"] == "pulsar")
    return f"pulsar {n_pul} / rfi {len(items) - n_pul}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25, help="每类支持集数量")
    ap.add_argument("--seed", type=int, default=0, help="支持集抽样种子")
    ap.add_argument("--rfi_query", type=int, default=2000, help="查询集 rfi 子采样数量")
    ap.add_argument("--query_seed", type=int, default=42, help="查询集固定种子（不随 seed 变）")
    ap.add_argument("--reserve", type=int, default=100, help="正类为支持池保留的样本数")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "region_gpps"))
    args = ap.parse_args()

    pool = load_pool()
    support, query = split(pool, args.k, args.seed, args.rfi_query,
                           args.query_seed, args.reserve)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "all.jsonl"), [to_region(s) for s in pool])
    write_jsonl(os.path.join(args.out, f"support_k{args.k}_s{args.seed}.jsonl"),
                [to_region(s) for s in support])
    write_jsonl(os.path.join(args.out, f"query_k{args.k}_s{args.seed}.jsonl"),
                [to_region(s) for s in query])
    print(f"GPPS all={len(pool)} ({label_counts(pool)})")
    print(f"support={len(support)} ({label_counts(support)}) "
          f"query={len(query)} ({label_counts(query)})")


if __name__ == "__main__":
    main()
