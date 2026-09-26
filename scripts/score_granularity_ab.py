"""A/B：整数票差 vs 连续分数，对 recall@低FPR 的影响。

背景：margin = (pulsar票数 − rfi票数) 只有 ~31 种取值 → ROC 只有 32 个台阶
      → recall@1%FPR 被量化扭曲。
本脚本用**同一批入选头、同一批缓存特征**，改成连续分数（每头 cosine(s1)−cosine(s0) 的均值），
对比两者的 recall@FPR。纯离线（CPU），不改任何运行中的实验。

用法: python scripts/score_granularity_ab.py --part results/l0_parts/k15_s0_h40.json \
        --support data/pulsar_gpps/support_k15_s0.jsonl --query data/pulsar_gpps/query_k15_s0.jsonl \
        --cache_dir features
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
torch.set_grad_enabled(False)

from src.utils import record_head_performance  # noqa: E402


def pick_heads(sup_feats, labels, num_heads):
    """镜像 run_baseline.evaluate_cached 的选头（全量头 → 逐类均值 → 逐头打分 → topk）。"""
    lab = list(labels)
    order = []
    for v in lab:
        if v not in order:
            order.append(v)
    masks = [torch.tensor([v == o for v in lab], dtype=torch.bool) for o in order]
    class_act = torch.stack([sup_feats[m].mean(0) for m in masks])
    sc = [0] * class_act.shape[1]
    for i in range(len(lab)):
        record_head_performance(class_act, sup_feats[i], order.index(lab[i]), sc)
    return np.argsort(np.array(sc))[-num_heads:][::-1].tolist(), class_act, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True)
    ap.add_argument('--support', required=True)
    ap.add_argument('--query', required=True)
    ap.add_argument('--cache_dir', default='features')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    info = json.load(open(args.part))
    H = info['num_heads']
    seg = os.path.basename(os.path.dirname(os.path.abspath(args.support)))
    pre = os.path.basename(args.support).replace('.jsonl', '')
    tag = args.tag or ('%s_%s' % (seg, pre))
    sup_f = os.path.join(args.cache_dir, tag + '_supfeat.pt')
    qry_f = os.path.join(args.cache_dir, tag + '_h%d_qryfeat.pt' % H)
    for p in (sup_f, qry_f):
        if not os.path.exists(p):
            print('缺缓存:', p); sys.exit(2)

    sup = torch.load(sup_f, map_location='cpu')
    qry = torch.load(qry_f, map_location='cpu')
    labs = np.array([1 if json.loads(l)['label'] == 'pulsar' else 0 for l in open(args.query)])
    sup_lab = [1 if json.loads(l)['label'] == 'pulsar' else 0 for l in open(args.support)]

    topk, class_act, order = pick_heads(sup, sup_lab, H)
    pos_i = order.index(1)
    # 每头质心（pulsar / rfi）
    c_pos = class_act[pos_i][topk]                       # [H,128]
    c_neg = class_act[1 - pos_i][topk]
    q = qry.float()                                       # [M,H,128]
    s_pos = torch.nn.functional.cosine_similarity(q, c_pos.unsqueeze(0), dim=-1)   # [M,H]
    s_neg = torch.nn.functional.cosine_similarity(q, c_neg.unsqueeze(0), dim=-1)
    cont = (s_pos - s_neg).mean(1).numpy()                # 连续分数
    v_cnt = (s_pos > s_neg).sum(1).numpy().astype(int)    # 票数（整数）
    votes = v_cnt - (H - v_cnt)                           # 票差 = json 里的 margin

    print('=== 分数粒度对比（同一批 %d 个头、同一批特征）===' % H)
    print('%-14s %-10s %-12s %-12s' % ('score', 'unique', 'AP', 'ROC点数'))
    for name, s in (('整数票差', votes), ('连续分数', cont)):
        ap_v = average_precision_score(labs, s)
        fpr, tpr, _ = roc_curve(labs, s)
        print('%-14s %-10d %-12.4f %-12d' % (name, len(np.unique(s)), ap_v, len(fpr)))
    print()
    print('%-10s %-14s %-14s %s' % ('FPR', '整数票差recall', '连续分数recall', '提升'))
    f1 = roc_curve(labs, votes); f2 = roc_curve(labs, cont)
    for t in (0.005, 0.01, 0.02, 0.05, 0.10):
        r1 = f1[1][f1[0] <= t].max() if (f1[0] <= t).any() else 0.0
        r2 = f2[1][f2[0] <= t].max() if (f2[0] <= t).any() else 0.0
        print('%-10s %-14.4f %-14.4f %+.4f' % ('%.1f%%' % (t * 100), r1, r2, r2 - r1))


if __name__ == '__main__':
    main()
