"""跑 SAVs 基线：k=20 头，每类 k 张支持，多随机种子，输出 F1/P/R mean±std。

用法:
  python scripts/run_baseline.py --k 25 --num_heads 20 --seeds 0,1,2,3,4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from tqdm import tqdm

torch.set_grad_enabled(False)

from metrics_util import confusion, pr_auc, recall_at_fpr, pr_curve  # noqa: E402
from src.utils import (load_model, mllm_encode, mllm_classify, mllm_classify_with_counts,
                        open_data, get_last_mean_head_activations, record_head_performance,
                        retrieve_examples_with_counts)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pulsar_gpps")


def evaluate(model, support_path, query_path, num_heads=20):
    train_data = open_data("pulsar", support_path)
    test_data = open_data("pulsar", query_path)
    print(f"\n  支持集 {len(train_data)} 张 -> 选 {num_heads} 头...")
    embeddings = mllm_encode(model, train_data, num_head=num_heads)
    print("  查询集分类中...")
    tp = fp = fn = 0
    _margins = []   # DUMP_SCORES_V1：每样本 (pulsar票数 - rfi票数)
    _finals = []
    for item in tqdm(test_data, desc="  分类"):
        pred, counts = mllm_classify_with_counts(item, model, embeddings)
        _margins.append(int(counts.get("pulsar", 0)) - int(counts.get("rfi", 0)))
        _finals.append(1 if pred == "pulsar" else 0)
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
    # DUMP_SCORES_V1 自检
    _tp = sum(1 for pr, sm in zip(_finals, test_data) if pr == 1 and sm["label"] == "pulsar")
    _fp = sum(1 for pr, sm in zip(_finals, test_data) if pr == 1 and sm["label"] != "pulsar")
    _fn = sum(1 for pr, sm in zip(_finals, test_data) if pr == 0 and sm["label"] == "pulsar")
    assert (_tp, _fp, _fn) == (tp, fp, fn), (
        "DUMP_SCORES_V1 自检失败: finals 反算 (%d,%d,%d) != 计数器 (%d,%d,%d)" % (_tp, _fp, _fn, tp, fp, fn))
    _labs = [1 if sm["label"] == "pulsar" else 0 for sm in test_data]
    _extra = {
        "confusion": confusion(_finals, _labs),
        "pr_auc": pr_auc(_margins, _labs),
        "recall_at_fpr": recall_at_fpr(_margins, _labs),
        "pr_curve": pr_curve(_margins, _labs),
        "n_pos": int(sum(_labs)),
        "n_neg": int(len(_labs) - sum(_labs)),
    }
    return {"f1": f1, "precision": p, "recall": r, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "n_query": len(test_data),
            "margins": _margins, "finals": _finals, **_extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_heads", type=int, default=20)
    ap.add_argument("--k", type=int, default=25, help="每类支持集数量")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--cache_dir", type=str, default=None, help="特征缓存目录")
    ap.add_argument("--data_dir", type=str, default=None, help="数据目录名（默认 pulsar_gpps）")
    # MULTIDATA_PATCH
    args = ap.parse_args()
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(
            os.path.dirname(__file__), "..", "data", args.data_dir)
        print("  DATA_DIR -> %s" % DATA_DIR)
    seeds = [int(x) for x in args.seeds.split(",")]

    model = load_model("qwen2.5_vl", "pulsar")
    all_results = {}
    for seed in seeds:
        sup = os.path.join(DATA_DIR, f"support_k{args.k}_s{seed}.jsonl")
        qry = os.path.join(DATA_DIR, f"query_k{args.k}_s{seed}.jsonl")
        if not os.path.exists(sup):
            print(f"缺少 {sup}，先运行 build_pulsar_data.py --k {args.k} --seed {seed}")
            continue
        print(f"\n=== seed {seed} ===")
        if args.cache_dir:
            tag = "pulsar_gpps_support_k%d_s%d" % (args.k, seed)
            r = evaluate_cached(model, sup, qry, args.num_heads, args.cache_dir, tag)
        else:
            r = evaluate(model, sup, qry, args.num_heads)
        all_results[f"s{seed}"] = r
        r = all_results[f"s{seed}"]
        print(f"  F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} Acc={r['accuracy']:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"baseline_k{args.k}_h{args.num_heads}_pulsar_gpps.json")
    with open(out, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n结果已保存: {out}")

    if len(all_results) > 1:
        import numpy as np
        f1s = np.array([r["f1"] for r in all_results.values()])
        ps = np.array([r["precision"] for r in all_results.values()])
        rs = np.array([r["recall"] for r in all_results.values()])
        print(f"\n== 汇总 (k={args.k}, heads={args.num_heads}, {len(all_results)} seeds) ==")
        print(f"F1       mean={f1s.mean():.4f} ± {f1s.std():.4f}")
        print(f"Precision mean={ps.mean():.4f} ± {ps.std():.4f}")
        print(f"Recall   mean={rs.mean():.4f} ± {rs.std():.4f}")


if __name__ == "__main__":
    main()


# ================= 整图特征缓存版（与四图 --cache_dir 对齐）=================


def _head_vecs(model, dataset, heads=None, desc="  提特征"):
    """每样本头向量；heads=None → 全量 784 头 [784,128]；否则只取指定头 [H,128]。"""
    out = []
    for item in tqdm(dataset, desc=desc):
        act = get_last_mean_head_activations([item], model, N_TRIALS=1, shot=0)
        if heads is None:
            out.append(act[:, :, -1, :].reshape(-1, 128))
        else:
            out.append(torch.stack([act[h[0], h[1], -1] for h in heads]))
    return torch.stack(out)


def evaluate_cached(model, support_path, query_path, num_heads=20, cache_dir=None, tag="pulsar_gpps"):
    """与 evaluate 同口径，但缓存支持/查询特征。"""
    train_data = open_data("pulsar", support_path)
    test_data = open_data("pulsar", query_path)
    os.makedirs(cache_dir, exist_ok=True)
    sup_cache = os.path.join(cache_dir, "%s_supfeat.pt" % tag)
    qry_cache = os.path.join(cache_dir, "%s_h%d_qryfeat.pt" % (tag, num_heads))

    if os.path.exists(sup_cache):
        sup_feats = torch.load(sup_cache)
        print("  加载支持集缓存: %s" % sup_cache)
    else:
        print("  支持集 %d 张 -> 提全量 784 头并缓存" % len(train_data))
        sup_feats = _head_vecs(model, train_data, None, "  支持集全量头")
        torch.save(sup_feats, sup_cache)
        print("  已缓存: %s" % sup_cache)

    # --- 选头：镜像 mllm_encode（逐类均值 → 逐头打分 → top-k）---
    seen, order = set(), []
    for it in train_data:
        if it["label"] not in seen:
            seen.add(it["label"]); order.append(it["label"])
    str_to_int = {l: i for i, l in enumerate(order)}
    int_to_str = {i: l for l, i in str_to_int.items()}
    lab = [str_to_int[it["label"]] for it in train_data]
    masks = [torch.tensor([v == str_to_int[l] for v in lab], dtype=torch.bool) for l in order]
    class_act = torch.stack([sup_feats[m].mean(0) for m in masks])
    success_count = [0] * class_act.shape[1]
    for i, it in enumerate(train_data):
        record_head_performance(class_act, sup_feats[i], str_to_int[it["label"]], success_count)
    arr = np.array(success_count)
    # 注意：不能把 [::-1] 的负步长视图直接喂给 torch 索引（ValueError），先转 list
    topk = np.argsort(arr)[-num_heads:][::-1].tolist()
    all_heads = model.all_heads
    top_heads = [all_heads[i] for i in topk]
    top_class_act = torch.stack([sup_feats[m][:, topk].mean(0) for m in masks])
    class_embed = {"activations": top_class_act, "top_heads": top_heads, "int_to_str": int_to_str}
    print("  选中 %d 头（镜像 mllm_encode 口径）" % len(top_heads))

    if os.path.exists(qry_cache):
        qf = torch.load(qry_cache)
        print("  加载查询集缓存: %s" % qry_cache)
    else:
        qf = _head_vecs(model, test_data, top_heads, "  查询集 top-头")
        torch.save(qf, qry_cache)
        print("  已缓存: %s" % qry_cache)

    tp = fp = fn = 0
    _margins, _finals = [], []
    for i, item in enumerate(tqdm(test_data, desc="  分类")):
        votes_by_index = retrieve_examples_with_counts(class_embed["activations"], qf[i])
        pred = class_embed["int_to_str"][votes_by_index[0][0]]
        counts = {class_embed["int_to_str"][ci]: c for ci, c in votes_by_index}
        _margins.append(int(counts.get("pulsar", 0)) - int(counts.get("rfi", 0)))
        _finals.append(1 if pred == "pulsar" else 0)
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

    _labs = [1 if sm["label"] == "pulsar" else 0 for sm in test_data]
    _tp = sum(1 for pr, y in zip(_finals, _labs) if pr == 1 and y == 1)
    _fp = sum(1 for pr, y in zip(_finals, _labs) if pr == 1 and y == 0)
    _fn = sum(1 for pr, y in zip(_finals, _labs) if pr == 0 and y == 1)
    assert (_tp, _fp, _fn) == (tp, fp, fn), (
        "缓存版自检失败: finals 反算 (%d,%d,%d) != 计数器 (%d,%d,%d)" % (_tp, _fp, _fn, tp, fp, fn))
    return {"f1": f1, "precision": p, "recall": r, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "n_query": len(test_data),
            "margins": _margins, "finals": _finals,
            "confusion": confusion(_finals, _labs),
            "pr_auc": pr_auc(_margins, _labs),
            "recall_at_fpr": recall_at_fpr(_margins, _labs),
            "pr_curve": pr_curve(_margins, _labs),
            "n_pos": int(sum(_labs)), "n_neg": int(len(_labs) - sum(_labs))}
