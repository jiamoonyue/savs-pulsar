"""头数消融（原论文 Figure 4 右）：固定支持集/查询集特征缓存，遍历 num_heads 做投票。

前提：已用 run_regionaware.py --store_full 生成全量头缓存：
  features/region_support_{k}_{seed}_{ptag}_fullqryfeat.pt   # [M, 4, 784, 128] fp16
  features/region_support_{k}_{seed}_{ptag}_supfeat.pt      # [N, 4, 784, 128]

纯 CPU（投票层秒级）。用法:
  python scripts/run_head_ablation.py --k 25 --seed 0 --prompt_type panel \
      --num_heads_list 5,10,15,20,30,40,50 --cache_dir features
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

torch.set_grad_enabled(False)

sys.path.insert(0, os.path.dirname(__file__))
from run_regionaware import (PANELS, build_centroids, classify_sample_v2,
                             region_head_scoring, select_heads)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "region")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate_num_heads(qry_feats_full, labels, support_feats, support_labels,
                       num_heads, q):
    """全量特征 [M,4,784,128] + 支持集 → 选头 → 投票 → F1/P/R。"""
    scores = region_head_scoring(support_feats,
                                 torch.tensor(support_labels, dtype=torch.long))
    heads, dominant = select_heads(scores, q=q, k=num_heads)
    centroids = build_centroids(support_feats,
                                torch.tensor(support_labels, dtype=torch.long),
                                heads, dominant)
    heads_list, domin = heads.tolist(), dominant.tolist()
    tp = fp = fn = 0
    for idx, y in enumerate(labels):
        full = qry_feats_full[idx]                  # [4, 784, 128] bf16（与正常路径同 dtype）
        feat = torch.stack([full[domin[j], h] for j, h in enumerate(heads_list)])
        votes, final, conf = classify_sample_v2(feat, heads, dominant, centroids)
        if y == 1:
            if final == 1:
                tp += 1
            else:
                fn += 1
        else:
            if final == 1:
                fp += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    acc = (tp + (len(labels) - tp - fp - fn)) / len(labels)
    return {"num_heads": num_heads, "f1": f1, "precision": p, "recall": r,
            "accuracy": acc, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25, help="支持集规模")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt_type", type=str, default="empty")
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--num_heads_list", type=str, default="5,10,15,20,30,40,50")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="region_gpps",
                    help="数据目录(region_gpps 纯GPPS/region 混数据)")
    args = ap.parse_args()
    heads_list_vals = [int(x) for x in args.num_heads_list.split(",")]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", args.data_dir)

    seg = args.data_dir            # 缓存前缀 = data_dir（region_gpps / region）
    cache_prefix = f"support_k{args.k}_s{args.seed}"
    base = os.path.join(args.cache_dir, f"{seg}_{cache_prefix}_{args.prompt_type}")
    sup_path = f"{base}_supfeat.pt"
    qry_path = f"{base}_fullqryfeat.pt"
    for p in (sup_path, qry_path):
        if not os.path.exists(p):
            print(f"缺少缓存 {p}，请先运行 --store_full 提取。")
            return

    support_feats = torch.load(sup_path)
    qry_feats_full = torch.load(qry_path)
    with open(os.path.join(data_dir, f"support_k{args.k}_s{args.seed}.jsonl")) as fh:
        support_labels = [0 if json.loads(l)["label"] == "rfi" else 1
                          for l in fh]
    with open(os.path.join(data_dir, f"query_k{args.k}_s{args.seed}.jsonl")) as fh:
        labels = [0 if json.loads(l)["label"] == "rfi" else 1 for l in fh]
    print(f"支持集 {support_feats.shape}, 查询集 {qry_feats_full.shape}")

    rows = []
    for nh in heads_list_vals:
        r = evaluate_num_heads(qry_feats_full, labels, support_feats,
                               support_labels, nh, args.q)
        print(f"  heads={nh:3d}: F1={r['f1']:.4f} P={r['precision']:.4f} "
              f"R={r['recall']:.4f} Acc={r['accuracy']:.4f}", flush=True)
        rows.append(r)

    out = os.path.join(OUT_DIR,
                       f"head_ablation_k{args.k}_s{args.seed}_{args.prompt_type}_{args.data_dir}.json")
    with open(out, "w") as fh:
        json.dump({"rows": rows, "pool": args.num_heads_list}, fh, indent=2)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
