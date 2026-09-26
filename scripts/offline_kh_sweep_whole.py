#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整图臂离线 K×H 扫描（MiraBest / FAST 整图通用）。

为什么需要它：
  四图臂有 offline_kh_sweep.py，但它复用 run_regionaware 的【区域感知选头】。
  整图臂的选头走 run_baseline.evaluate_cached 的「镜像 mllm_encode 口径」——
  算法不同，不能加分支，只能另写。

原理：
  查询集不随 k/seed 变 → 查询侧【全量 784 头】缓存对所有 (k, seed, H) 通用。
  支持集缓存按 (k, seed) 各有，且已是全量 784 头。
  选头 + 分类全程纯 CPU。

口径保证：
  直接 import src.utils 的 record_head_performance / retrieve_examples_with_counts /
  open_data，不另写一套；dtype 保持 bf16（转 fp32 会翻转边界判断，见 HANDOFF 坑 13）。
  success_count 只依赖 (k, seed)（与 H 无关），故外层按 (k, seed) 算一次后按 H 复用。

等价性已实测：
  qry_full[:, topk] 与管线 _head_vecs(..., top_heads) 的 h40 缓存【逐位相同】
  （max abs diff = 0.0, 0/2406400 不等）→ 本脚本结果应等于管线值，无口径差。

用法：
  python -u scripts/offline_kh_sweep_whole.py \
    --sup-cache features_mira --sup-tag pulsar_gpps \
    --qry-cache 'features_mira_full/MiraBest数据集_support_fullqryfeat.pt' \
    --data-root '/SAVS_DATA_ROOT/CCF-A/MiraBest数据集/whole' \
    --ks 5,15,25,50 --heads 20,40,80,120,200 --seeds 0,1,2,3,4
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

import numpy as np          # noqa: E402
import torch                # noqa: E402

import metrics_util as M    # noqa: E402

torch.set_grad_enabled(False)

from src.utils import (record_head_performance, retrieve_examples_with_counts,
                       open_data)  # noqa: E402


def head_scores(sup_feats, sup_data):
    """逐位复刻 run_baseline.evaluate_cached 的选头打分部分。与 H 无关。"""
    seen, order = set(), []
    for it in sup_data:
        if it["label"] not in seen:
            seen.add(it["label"])
            order.append(it["label"])
    str_to_int = {l: i for i, l in enumerate(order)}
    int_to_str = {i: l for l, i in str_to_int.items()}
    lab = [str_to_int[it["label"]] for it in sup_data]
    masks = [torch.tensor([v == str_to_int[l] for v in lab], dtype=torch.bool) for l in order]
    class_act = torch.stack([sup_feats[m].mean(0) for m in masks])
    success_count = [0] * class_act.shape[1]
    for i, it in enumerate(sup_data):
        record_head_performance(class_act, sup_feats[i], str_to_int[it["label"]], success_count)
    return np.array(success_count), masks, int_to_str


def take_top(arr, masks, sup_feats, int_to_str, H):
    """逐位复刻 evaluate_cached 的 top-k 选取与类质心构建。"""
    topk = np.argsort(arr)[-H:][::-1].tolist()
    top_class_act = torch.stack([sup_feats[m][:, topk].mean(0) for m in masks])
    return topk, top_class_act, int_to_str


def classify_all(top_class_act, int_to_str, qf):
    """逐位复刻 evaluate_cached 的分类循环。qf: [M, H, 128] -> (finals, margins)。

    margins = pulsar 票 − rfi 票，与管线 run_baseline 的 margin 口径一致，
    供 AP / recall@1%FPR 使用（F1/P/R 仍走硬预测 finals）。
    """
    pos_i = [k for k, v in int_to_str.items() if v == "pulsar"][0]
    neg_i = [k for k, v in int_to_str.items() if v == "rfi"][0]
    finals, margins = [], []
    for i in range(qf.shape[0]):
        votes = retrieve_examples_with_counts(top_class_act, qf[i])
        counts = dict(votes)
        pred = int_to_str[votes[0][0]]
        finals.append(1 if pred == "pulsar" else 0)
        margins.append(counts.get(pos_i, 0) - counts.get(neg_i, 0))
    return finals, margins


def prf(finals, labs):
    tp = sum(1 for p, y in zip(finals, labs) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(finals, labs) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(finals, labs) if p == 0 and y == 1)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return f1, p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sup-cache', required=True, help='支持集缓存目录')
    ap.add_argument('--sup-tag', required=True, help='支持集缓存前缀（如 pulsar_gpps）')
    ap.add_argument('--qry-cache', required=True, help='查询集全量头缓存文件 (.pt)')
    ap.add_argument('--data-root', required=True, help='jsonl 根目录')
    ap.add_argument('--ks', default='5,15,25,50')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--heads', default='20,40,80,120,200')
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    Hs = [int(x) for x in args.heads.split(',')]

    if not os.path.exists(args.qry_cache):
        print('!! 找不到查询集全量缓存: %s' % args.qry_cache)
        sys.exit(2)
    qry = torch.load(args.qry_cache, map_location='cpu')
    print('查询缓存: %s  %s  %s' % (os.path.basename(args.qry_cache), tuple(qry.shape), qry.dtype))
    if qry.dim() != 3:
        print('!! 期望整图 3 维 [M, 784, 128]，实际 %d 维 —— 整图扫描器不接受面板维' % qry.dim())
        sys.exit(2)

    qpath = os.path.join(args.data_root, 'query_k%d_s%d.jsonl' % (ks[0], seeds[0]))
    ql = np.array([1 if it["label"] == "pulsar" else 0 for it in open_data("pulsar", qpath)])
    assert len(ql) == qry.shape[0], '标签数 %d != 查询特征数 %d' % (len(ql), qry.shape[0])
    print('查询集 %d 张（pos %d / neg %d）\n' % (len(ql), int(ql.sum()), int(len(ql) - ql.sum())))

    print('%-6s %-6s %-7s %-17s %-17s %-17s %-17s %-17s'
          % ('k', 'H', 'n_seed', 'F1(mean±std)', 'P(mean±std)', 'R(mean±std)',
             'AP(mean±std)', 'r@1%FPR(mean±std)'))
    print('-' * 110)

    results = {}
    missing = []
    for k in ks:
        for s in seeds:
            sup_path = os.path.join(
                args.sup_cache, '%s_support_k%d_s%d_supfeat.pt' % (args.sup_tag, k, s))
            if not os.path.exists(sup_path):
                missing.append((k, s))
                continue
            sup_feats = torch.load(sup_path, map_location='cpu')
            sup_data = open_data(
                "pulsar", os.path.join(args.data_root, 'support_k%d_s%d.jsonl' % (k, s)))
            arr, masks, int_to_str = head_scores(sup_feats, sup_data)
            for H in Hs:
                topk, tca, i2s = take_top(arr, masks, sup_feats, int_to_str, H)
                qf = qry[:, topk]
                finals, margins = classify_all(tca, i2s, qf)
                results.setdefault((k, H), []).append(
                    prf(finals, ql) + (M.pr_auc(margins, ql),
                                       M.recall_at_fpr(margins, ql)['0.01']))
            print('  [done] k=%d seed=%d' % (k, s), flush=True)

    if missing:
        print('  !! 缺支持集缓存: %s' % missing)

    results_out = []
    for k in ks:
        for H in Hs:
            res = results.get((k, H), [])
            if not res:
                print('%-6d %-6d %-7s %s' % (k, H, '0', '（无支持集缓存）'))
                continue
            f1s = np.array([r[0] for r in res])
            ps = np.array([r[1] for r in res])
            rs = np.array([r[2] for r in res])
            aps = np.array([r[3] for r in res])
            rafs = np.array([r[4] for r in res])
            line = ('%-6d %-6d %-7d %-17s %-17s %-17s %-17s %-17s'
                    % (k, H, len(res),
                       '%.4f±%.4f' % (f1s.mean(), f1s.std()),
                       '%.4f±%.4f' % (ps.mean(), ps.std()),
                       '%.4f±%.4f' % (rs.mean(), rs.std()),
                       '%.4f±%.4f' % (aps.mean(), aps.std()),
                       '%.4f±%.4f' % (rafs.mean(), rafs.std())))
            print(line)
            results_out.append((k, H, f1s.mean()))

    if results_out:
        bk, bH, bf = max(results_out, key=lambda x: x[2])
        print('\n>>> 本表最优 F1: k=%d H=%d F1=%.4f' % (bk, bH, bf))


if __name__ == '__main__':
    main()
