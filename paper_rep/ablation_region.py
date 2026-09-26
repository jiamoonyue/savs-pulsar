# -*- coding: utf-8 -*-
"""FAST hard 消融：把 region-aware 的三个组成部分拆开验证。
所有变体直接 import run_regionaware 的函数，评测口径与管线一致（同缓存、同划分、同种子）。

自校验：变体 "routing"（完整方法）应复现 RESULTS.md 的 0.7797±0.0088。
"""
import json
import sys

import numpy as np
import torch

HERE = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/scripts"
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/..")
torch.set_grad_enabled(False)
import run_regionaware as RA  # noqa: E402

CACHE = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/features"
DATA = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/data/region_gpps"
K, H, Q = 50, 40, 2
RA.PANELS = ["freq_phase", "time_phase", "dm_curve", "profile"]


def f1_pr(score, lab):
    pred = (score > 0).astype(int)
    tp = int(((pred == 1) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f, p, r


def cents_for(sup, lab, h, r):
    """与 RA.build_centroids 相同的质心算法，但面板 r 可指定。"""
    fr = sup[:, r, h]
    return fr[lab == 0].mean(0), fr[lab == 1].mean(0)


def vote(qry, h_idx, panels_per_head, cents):
    """每头在其 panels_per_head[j] 列出的每个面板上各投一票，多数决定。"""
    N = qry.shape[0]
    out = np.zeros(N, dtype=np.int64)
    hl = h_idx.tolist() if hasattr(h_idx, "tolist") else list(h_idx)
    for i in range(N):
        n1 = 0
        total = 0
        for j, h in enumerate(hl):
            for r in panels_per_head[j]:
                c0, c1 = cents[(h, r)]
                v = qry[i, r, h]
                s0 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c0.unsqueeze(0))
                s1 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c1.unsqueeze(0))
                n1 += 1 if s1 > s0 else 0
                total += 1
        out[i] = n1 - (total - n1)
    return out


def classify_routing(qry, h_idx, dom, cents):
    """与 offline_kh_sweep.vote_all 相同：每头只在其主导面板投一票。"""
    N = qry.shape[0]
    out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        votes = []
        for j, h in enumerate(h_idx.tolist()):
            c0, c1 = cents[h]
            v = qry[i, int(dom[j]), h]
            s0 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c0.unsqueeze(0))
            s1 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c1.unsqueeze(0))
            votes.append(1 if s1 > s0 else 0)
        n1 = sum(votes)
        out[i] = n1 - (len(votes) - n1)
    return out


res = {}
QRY_FULL = "%s/region_gpps_support_k25_s0_empty_fullqryfeat.pt" % CACHE
qry = torch.load(QRY_FULL, map_location="cpu").float()   # [2896, 4, 784, 128]，查询集固定
print("全量查询缓存:", tuple(qry.shape), flush=True)
for s in range(5):
    sup = torch.load("%s/region_gpps_support_k%d_s%d_empty_supfeat.pt" % (CACHE, K, s),
                     map_location="cpu").float()
    lab_s = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                      for l in open("%s/support_k%d_s%d.jsonl" % (DATA, K, s))])
    lab_q = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                      for l in open("%s/query_k%d_s%d.jsonl" % (DATA, K, s))])
    tlab = torch.tensor(lab_s)

    scores = RA.region_head_scoring(sup, tlab, mode="insample")  # [784, 4]

    # (a) 完整 region-aware 路由（自校验：离线口径应复现离线 K×H 表的 k=50/H=40 = 0.7521±0.0392）
    h_idx, dom = RA.select_heads(scores, q=Q, k=H)
    cents = RA.build_centroids(sup, tlab, h_idx, dom)
    sc = classify_routing(qry, h_idx, dom, cents)
    res.setdefault("routing", []).append(f1_pr(sc, lab_q))

    # (b) 面板分解 + 逐面板打分，但【不做主导面板路由】：每头在全部 4 个面板各投一票
    cents_all = {(h, r): cents_for(sup, tlab, h, r) for h in h_idx.tolist() for r in range(4)}
    pph = [list(range(4)) for _ in h_idx.tolist()]
    sc = vote(qry, h_idx, pph, cents_all)
    res.setdefault("no_routing", []).append(f1_pr(sc, lab_q))

    # (c) 单面板：只用该面板打分与分类
    for pi, pname in enumerate(RA.PANELS):
        h1 = scores[:, pi].argsort(descending=True)[:H]
        c1 = {(h, pi): cents_for(sup, tlab, h, pi) for h in h1.tolist()}
        sc = vote(qry, h1.tolist(), [[pi]] * len(h1), c1)
        res.setdefault("panel_" + pname, []).append(f1_pr(sc, lab_q))

    # (d) freq_phase + time_phase 两面板（打分与路由都限制在这两个面板内）
    st = scores[:, [0, 1]]
    h2, dom2 = RA.select_heads(st, q=Q, k=H)
    dom2 = torch.tensor([0 if int(d) == 0 else 1 for d in dom2.tolist()])
    c2 = {h: cents_for(sup, tlab, h, [0, 1][int(dom2[j])]) for j, h in enumerate(h2.tolist())}
    sc = classify_routing(qry[:, [0, 1]], h2, dom2, c2)
    res.setdefault("freq+time", []).append(f1_pr(sc, lab_q))

    # (e) 完整方法但头打分改用 leave-one-out（审稿人第 2 条：selection bias）
    scores_loo = RA.region_head_scoring(sup, tlab, mode="loo")
    h3, d3 = RA.select_heads(scores_loo, q=Q, k=H)
    c3 = RA.build_centroids(sup, tlab, h3, d3)
    sc = classify_routing(qry, h3, d3, c3)
    res.setdefault("routing_loo", []).append(f1_pr(sc, lab_q))

    print("seed %d done" % s, flush=True)

print()
print("%-28s %-18s %-18s %-18s" % ("变体", "F1(mean±std)", "P(mean±std)", "R(mean±std)"))
print("-" * 90)
for name in ["routing", "no_routing", "routing_loo", "freq+time",
             "panel_freq_phase", "panel_time_phase", "panel_dm_curve", "panel_profile"]:
    if name not in res:
        continue
    a = np.array([x[0] for x in res[name]])
    b = np.array([x[1] for x in res[name]])
    c = np.array([x[2] for x in res[name]])
    print("%-28s %-18s %-18s %-18s"
          % (name, "%.4f±%.4f" % (a.mean(), a.std()),
             "%.4f±%.4f" % (b.mean(), b.std()), "%.4f±%.4f" % (c.mean(), c.std())))
