"""从 features/ 缓存离线重算每样本投票 margin，产出 PR-AUC / recall@FPR / 混淆矩阵。

为什么需要：结果 json 只存聚合计数（tp/fp/fn），**不含每样本分数**，
因此 PR-AUC、recall@FPR 无法从 json 直接算。四图的特征缓存里有原料：
  - 支持集全量头缓存 [N,4,784,128]  → 重建类质心
  - 查询集缓存 [M,k,128]（入选头在其主导面板上的向量）
  - 结果 json 的 heads 字段（layer/head/dominant_panel）
  → 用模块里**真实的** build_centroids / classify_sample_v2 重算，不另写一套。

自检（硬闸门）：重算得到的 tp/fp/fn 必须与结果 json 里存的完全一致，
否则说明重算口径与正式实现不一致 → 直接 FAIL 退出，不给数字。

用法:
  python scripts/pr_metrics.py --json results/region_parts/<part>.json \
      --support data/region_gpps/support_k15_s2.jsonl \
      --query   data/region_gpps/query_k15_s2.jsonl \
      --cache_dir features
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

torch.set_grad_enabled(False)

from run_regionaware import PANELS, build_centroids, classify_sample_v2  # noqa: E402


def pr_auc(scores, labels):
    """Average Precision（PR 曲线下面积），用梯形/逐步法。labels: 1=正类。"""
    order = np.argsort(-scores, kind='stable')
    y = np.asarray(labels, dtype=float)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = y.sum()
    if n_pos == 0:
        return float('nan')
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / n_pos
    # 逐步 AP（sklearn average_precision 口径）：每个正类召回增量 × 当时精度
    ap = 0.0
    prev_rec = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            ap += (rec[i] - prev_rec) * prec[i]
            prev_rec = rec[i]
    return float(ap)


def recall_at_fpr(scores, labels, fpr_targets=(0.01, 0.05, 0.10, 0.20)):
    """在给定假阳率（相对负类总数）下能达到的召回。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    out = {}
    for t in fpr_targets:
        allowed_fp = int(np.floor(t * neg))
        order = np.argsort(-s, kind='stable')
        ys = y[order]
        cum_fp = np.cumsum(1 - ys)
        ok = cum_fp <= allowed_fp
        if not ok.any():
            out[t] = 0.0
            continue
        k = np.max(np.where(ok)[0])
        out[t] = float(np.cumsum(ys)[k] / pos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--support', required=True)
    ap.add_argument('--query', required=True)
    ap.add_argument('--cache_dir', default='features')
    args = ap.parse_args()

    info = json.load(open(args.json))
    heads_info = info['heads']
    h_idx = torch.tensor([hi['layer'] * 28 + hi['head'] for hi in heads_info])
    dom = torch.tensor([PANELS.index(hi['dominant_panel']) for hi in heads_info])

    seg = os.path.basename(os.path.dirname(os.path.abspath(args.support)))       # region_gpps
    sup_pre = os.path.basename(args.support).replace('.jsonl', '')               # support_k15_s2
    qry_pre = os.path.basename(args.query).replace('.jsonl', '')
    ptag = info.get('prompt_type', 'empty')
    h, q = info['num_heads'], info['q']
    sup_cache = os.path.join(args.cache_dir, '%s_%s_%s_supfeat.pt' % (seg, sup_pre, ptag))
    # 注意：run_regionaware 里 query 缓存的前缀用的是【支持集】文件名
    # （cache_prefix = basename(support_path)），不是 query 文件名。
    qry_cache = os.path.join(args.cache_dir, '%s_%s_%s_h%d_q%d_qryfeat.pt' % (seg, sup_pre, ptag, h, q))
    for p in (sup_cache, qry_cache):
        if not os.path.exists(p):
            print('缺少缓存: %s' % p); sys.exit(2)

    # ⚠ 必须与官方实现同 dtype：全程 bf16（DEV_PROTOCOL §4「投票全程 bf16」）。
    # 若这里 .float()，bf16→fp32 的舍入会翻转边界处的 s1>s0 判断，
    # 导致重算的 tp/fp/fn 与 json 不一致（实测会 MISMATCH）。
    sup_feats = torch.load(sup_cache, map_location='cpu')
    qry_feats = torch.load(qry_cache, map_location='cpu')
    sup_rows = [json.loads(l) for l in open(args.support)]
    qry_rows = [json.loads(l) for l in open(args.query)]
    sup_lab = [0 if r['label'] == 'rfi' else 1 for r in sup_rows]
    qry_lab = [0 if r['label'] == 'rfi' else 1 for r in qry_rows]

    centroids = build_centroids(sup_feats, torch.tensor(sup_lab), h_idx, dom)

    margins, preds = [], []
    for i in range(qry_feats.shape[0]):
        votes, final, _ = classify_sample_v2(qry_feats[i], h_idx, dom, centroids, None, info.get('method', 'a1'))
        margins.append(int(sum(votes) - (len(votes) - sum(votes))))
        preds.append(int(final))

    tp = sum(1 for p, y in zip(preds, qry_lab) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, qry_lab) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, qry_lab) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, qry_lab) if p == 0 and y == 0)

    print('=== 自检：重算 vs json 存的 ===')
    ok = (tp == info['tp'] and fp == info['fp'] and fn == info['fn'])
    print('  重算 tp/fp/fn = %d/%d/%d     json = %d/%d/%d   %s'
          % (tp, fp, fn, info['tp'], info['fp'], info['fn'], 'MATCH' if ok else 'MISMATCH'))
    if not ok:
        print('  -> 重算口径与正式实现不一致，拒绝给数字'); sys.exit(1)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print('  由混淆矩阵：P=%.4f R=%.4f F1=%.4f (json F1=%.4f)' % (P, R, F1, info['f1']))

    print('\n=== 不平衡敏感指标（单点 F1 之外的）===')
    print('  混淆矩阵: TP=%d FP=%d FN=%d TN=%d' % (tp, fp, fn, tn))
    print('  多数类基线 accuracy = %.4f  （全判 rfi 就有这个数，故 accuracy 无信息量）'
          % (qry_lab.count(0) / len(qry_lab)))
    ap_val = pr_auc(np.array(margins), np.array(qry_lab))
    print('  PR-AUC (Average Precision) = %.4f   ← 对负类数量不敏感' % ap_val)
    raf = recall_at_fpr(margins, qry_lab)
    for t, v in raf.items():
        print('  recall@FPR=%.0f%%  →  %.4f' % (t * 100, v))
    print('  margin 分布（正/负类）: 正均值 %.2f, 负均值 %.2f'
          % (np.mean([m for m, y in zip(margins, qry_lab) if y == 1]),
             np.mean([m for m, y in zip(margins, qry_lab) if y == 0])))


if __name__ == '__main__':
    main()
