"""单进程串行跑全部单面板实验：模型只加载一次，循环 (panel, k, seed) 组合。

背景：多进程并行时每个进程都加载模型（fp32 36GB -> 峰值 ~90GB/进程），
共享机（load 高 + 资源竞争）下多次被 OOM 杀。此脚本单进程只加载一次，
峰值内存 ~90GB 只会发生一次，串行跑完所有组合，最稳。

用法:
  python scripts/run_single_panel_all.py --panels freq_phase,time_phase \
      --ks 5,10,25,50 --seeds 0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

torch.set_grad_enabled(False)

from src.utils import load_model, mllm_encode, mllm_classify

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "region_gpps")
PANELS = ["freq_phase", "time_phase", "dm_curve", "profile"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.dirname(__file__))
from run_single_panel import to_panel_jsonl, load_jsonl


def evaluate(model, support, query, num_heads, panel):
    train_data = to_panel_jsonl(support, panel)
    test_data = to_panel_jsonl(query, panel)
    print(f"  [k={''} 选 {num_heads} 头]")
    embeddings = mllm_encode(model, train_data, num_head=num_heads)
    tp = fp = fn = 0
    for item in test_data:
        pred = mllm_classify(item, model, embeddings)
        if item["label"] == "pulsar":
            if pred == "pulsar":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "pulsar":
                fp += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    acc = (tp + (len(test_data) - tp - fp - fn)) / len(test_data)
    return {"f1": f1, "precision": p, "recall": r, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "n_query": len(test_data)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=str, default=",".join(PANELS))
    ap.add_argument("--ks", type=str, default="25", help="逗号分隔的支持集规模")
    ap.add_argument("--num_heads", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="0")
    args = ap.parse_args()
    panels = [x for x in args.panels.split(",") if x]
    ks = [int(x) for x in args.ks.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",")]
    total = len(panels) * len(ks) * len(seeds)

    print(f"加载模型（一次，峰值内存 ~90GB）...")
    model = load_model("qwen2.5_vl", "pulsar")
    print(f"模型就绪。开始 {total} 个实验（串行）")

    n = 0
    for panel in panels:
        for k in ks:
            for seed in seeds:
                n += 1
                sup = os.path.join(DATA_DIR, f"support_k{k}_s{seed}.jsonl")
                qry = os.path.join(DATA_DIR, f"query_k{k}_s{seed}.jsonl")
                if not os.path.exists(sup):
                    print(f"[{n}/{total}] 缺少 {sup}，跳过")
                    continue
                print(f"\n[{n}/{total}] panel={panel} k={k} seed={seed} ...", flush=True)
                support = load_jsonl(sup, 0)
                query = load_jsonl(qry, 0)
                res = evaluate(model, support, query, args.num_heads, panel)
                print(f"  [{panel} k={k} s={seed}] F1={res['f1']:.4f} "
                      f"P={res['precision']:.4f} R={res['recall']:.4f} "
                      f"Acc={res['accuracy']:.4f}", flush=True)
                os.makedirs(OUT_DIR, exist_ok=True)
                out = os.path.join(OUT_DIR, f"panel_{panel}_k{k}_h{args.num_heads}_region_gpps.json")
                with open(out, "w") as fh:
                    json.dump({f"s{seed}": res}, fh, indent=2)

    print("\n=== 全部完成，结果汇总 ===")
    for panel in panels:
        for k in ks:
            vals = []
            for seed in seeds:
                fn_j = os.path.join(OUT_DIR, f"panel_{panel}_k{k}_h{args.num_heads}_region_gpps.json")
                if os.path.exists(fn_j):
                    d = json.load(open(fn_j))
                    vals.append(d.get(f"s{seed}", {}).get("f1", 0))
            if vals:
                print(f"  {panel:11s} k={k:2d}: F1={np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
