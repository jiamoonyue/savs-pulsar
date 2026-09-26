"""单面板 SAVs 验证：只用指定面板子图做输入，跑官方 SAVs 流程（全局评分+top-k+多数投票）。

目的：验证"子图独立前向是否有判别力"（区域感知的核心假设），
并得到每面板的独立判别力（论文分析素材）。

用法:
  python scripts/run_single_panel.py --panel freq_phase --k 25 --num_heads 20 --seeds 0 --limit 0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

torch.set_grad_enabled(False)

from tqdm import tqdm

from src.utils import load_model, mllm_encode, mllm_classify, open_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "region")
PANELS = ["freq_phase", "time_phase", "dm_curve", "profile"]


def to_panel_jsonl(items, panel):
    """把 region 样本（panels dict）转成单图 jsonl 列表（image=指定面板）。"""
    return [{"image": it["panels"][panel], "question": "", "label": it["label"]} for it in items]


def evaluate(model, support, query, num_heads, panel):
    train_data = to_panel_jsonl(support, panel)
    test_data = to_panel_jsonl(query, panel)
    embeddings = mllm_encode(model, train_data, num_head=num_heads)
    print("  查询集分类中...", flush=True)
    tp = fp = fn = 0
    for item in tqdm(test_data, desc="  分类"):
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


def load_jsonl(path, limit=0):
    with open(path) as fh:
        items = [json.loads(line) for line in fh]
    return items[:limit] if limit > 0 else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, required=True, choices=PANELS)
    ap.add_argument("--num_heads", type=int, default=20)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评估前 N 个查询样本")
    ap.add_argument("--data_dir", type=str, default="region_gpps", help="数据目录: region_gpps(纯GPPS,默认) / region(混数据)")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", args.data_dir)

    model = load_model("qwen2.5_vl", "pulsar")
    all_results = {}
    for seed in seeds:
        sup = os.path.join(data_dir, f"support_k{args.k}_s{seed}.jsonl")
        qry = os.path.join(data_dir, f"query_k{args.k}_s{seed}.jsonl")
        if not os.path.exists(sup):
            print(f"缺少 {sup}，先运行 build_{args.data_dir}_data.py --k {args.k} --seed {seed}")
            continue
        print(f"\n=== seed {seed} ===")
        support = load_jsonl(sup, 0)
        query = load_jsonl(qry, args.limit)
        all_results[f"s{seed}"] = evaluate(model, support, query, args.num_heads, args.panel)
        r = all_results[f"s{seed}"]
        print(f"  [{args.panel}] F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} Acc={r['accuracy']:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"panel_{args.panel}_k{args.k}_h{args.num_heads}_{args.data_dir}.json")
    with open(out, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n结果已保存: {out}")
    if len(all_results) > 1:
        f1s = np.array([r["f1"] for r in all_results.values()])
        ps = np.array([r["precision"] for r in all_results.values()])
        rs = np.array([r["recall"] for r in all_results.values()])
        print(f"== 汇总 [{args.panel}] (k={args.k}, {len(all_results)} seeds) ==")
        print(f"F1 mean={f1s.mean():.4f} ± {f1s.std():.4f} | P mean={ps.mean():.4f} | R mean={rs.mean():.4f}")


if __name__ == "__main__":
    main()
