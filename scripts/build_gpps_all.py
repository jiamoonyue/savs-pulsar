"""批量生成 GPPS 整图 / 四图数据（统一划分，一次加载 pool）。

两个输出目录由同一 split 派生 → 同 (k, seed) 下支持/查询集逐样本相同。
all.jsonl 也一并刷新（整图 / 四图两种格式）。

用法:
  python scripts/build_gpps_all.py --ks 5,15,25,50 --seeds 0,1,2,3,4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpps_data import load_pool, split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_WHOLE = os.path.join(HERE, "..", "data", "pulsar_gpps")
DEF_REGION = os.path.join(HERE, "..", "data", "region_gpps")


def to_whole(s):
    return {"image": s["whole"], "question": "", "label": s["label"]}


def to_region(s):
    return {"uid": s["uid"], "label": s["label"], "panels": s["panels"]}


def write_jsonl(path, items):
    with open(path, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


def label_counts(items):
    n = sum(1 for s in items if s["label"] == "pulsar")
    return f"pulsar {n} / rfi {len(items) - n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="5,15,25,50", help="逗号分隔的 k 列表")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="逗号分隔的 seed 列表")
    ap.add_argument("--rfi_query", type=int, default=2000)
    ap.add_argument("--query_seed", type=int, default=42)
    ap.add_argument("--reserve", type=int, default=100)
    ap.add_argument("--whole_out", default=DEF_WHOLE)
    ap.add_argument("--region_out", default=DEF_REGION)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    pool = load_pool()
    print(f"pool={len(pool)} ({label_counts(pool)})")
    os.makedirs(args.whole_out, exist_ok=True)
    os.makedirs(args.region_out, exist_ok=True)
    write_jsonl(os.path.join(args.whole_out, "all.jsonl"), [to_whole(s) for s in pool])
    write_jsonl(os.path.join(args.region_out, "all.jsonl"), [to_region(s) for s in pool])

    for k in ks:
        for seed in seeds:
            support, query = split(pool, k, seed, args.rfi_query,
                                   args.query_seed, args.reserve)
            write_jsonl(os.path.join(args.whole_out, f"support_k{k}_s{seed}.jsonl"),
                        [to_whole(s) for s in support])
            write_jsonl(os.path.join(args.whole_out, f"query_k{k}_s{seed}.jsonl"),
                        [to_whole(s) for s in query])
            write_jsonl(os.path.join(args.region_out, f"support_k{k}_s{seed}.jsonl"),
                        [to_region(s) for s in support])
            write_jsonl(os.path.join(args.region_out, f"query_k{k}_s{seed}.jsonl"),
                        [to_region(s) for s in query])
        print(f"  k={k} done (support={label_counts(support)} query={label_counts(query)})")
    print("ALL DONE")


if __name__ == "__main__":
    main()
