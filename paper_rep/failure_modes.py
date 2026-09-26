# -*- coding: utf-8 -*-
"""失效模式分析（FAST hard, seed 0, k=50, N=40, 完整 region-aware 路由）。
定量刻画 FN/FP：用 textinfo 的 chi2 / period / dm 检验"弱信号是主要失效模式"。"""
import collections
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
TI = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/data/textinfo/textinfo_all_k25.json"
K, H, Q, S = 50, 40, 2, 0

qry = torch.load("%s/region_gpps_support_k25_s0_empty_fullqryfeat.pt" % CACHE,
                 map_location="cpu").float()
sup = torch.load("%s/region_gpps_support_k%d_s%d_empty_supfeat.pt" % (CACHE, K, S),
                 map_location="cpu").float()
lab_s = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                  for l in open("%s/support_k%d_s%d.jsonl" % (DATA, K, S))])
rows = [json.loads(l) for l in open("%s/query_k%d_s%d.jsonl" % (DATA, K, S))]
lab_q = np.array([1 if r["label"] == "pulsar" else 0 for r in rows])
uids = [r["uid"] for r in rows]
ti = json.load(open(TI))

scores = RA.region_head_scoring(sup, torch.tensor(lab_s), mode="insample")
h_idx, dom = RA.select_heads(scores, q=Q, k=H)
cents = RA.build_centroids(sup, torch.tensor(lab_s), h_idx, dom)

N = qry.shape[0]
pred = np.zeros(N, dtype=np.int64)
for i in range(N):
    v = []
    for j, h in enumerate(h_idx.tolist()):
        c0, c1 = cents[h]
        x = qry[i, int(dom[j]), h]
        s0 = torch.nn.functional.cosine_similarity(x.unsqueeze(0), c0.unsqueeze(0))
        s1 = torch.nn.functional.cosine_similarity(x.unsqueeze(0), c1.unsqueeze(0))
        v.append(1 if s1 > s0 else 0)
    pred[i] = 1 if sum(v) * 2 > len(v) else 0

tp = np.where((pred == 1) & (lab_q == 1))[0]
fp = np.where((pred == 1) & (lab_q == 0))[0]
fn = np.where((pred == 0) & (lab_q == 1))[0]
tn = np.where((pred == 0) & (lab_q == 0))[0]
print("TP=%d FP=%d FN=%d TN=%d" % (len(tp), len(fp), len(fn), len(tn)))
P = len(tp) / (len(tp) + len(fp))
R = len(tp) / (len(tp) + len(fn))
print("P=%.4f R=%.4f F1=%.4f" % (P, R, 2 * P * R / (P + R)))

chi = {u: float(d["chi2"]) for u, d in ti.items() if "chi2" in d}
print("\n--- 失效模式：真脉冲星的 chi2（信噪比代理）---")
for name, idx in [("TP", tp), ("FN", fn)]:
    v = [chi.get(uids[i]) for i in idx if uids[i] in chi]
    v = np.array(sorted(v))
    print("%s: n=%d  中位 chi2=%.1f  P25=%.1f  P10=%.1f"
          % (name, len(v), np.median(v), np.percentile(v, 25), np.percentile(v, 10)))

tp_med = np.median([chi.get(uids[i]) for i in tp if uids[i] in chi])
fn_vals = [chi.get(uids[i]) for i in fn if uids[i] in chi]
q10 = np.percentile([chi.get(uids[i]) for i in tp if uids[i] in chi], 10)
print("FN 中 chi2 低于 TP 十分位的比例: %.1f%%"
      % (100 * sum(1 for x in fn_vals if x < q10) / max(len(fn_vals), 1)))

print("\n--- FN 样例（uid, chi2, period, dm）---")
fn_sorted = sorted(fn, key=lambda i: chi.get(uids[i], 1e9))
for i in fn_sorted[:6]:
    d = ti.get(uids[i], {})
    print("  %s  chi2=%s  period=%s  dm=%s" % (uids[i], d.get("chi2"), d.get("period"), d.get("dm")))

print("\n--- FP 样例（uid, chi2, period, dm, label）---")
for i in fp[:6]:
    d = ti.get(uids[i], {})
    print("  %s  chi2=%s  period=%s  dm=%s  label=%s"
          % (uids[i], d.get("chi2"), d.get("period"), d.get("dm"), d.get("label")))

print("\n--- RFI/脉冲星的 (period,dm) 指纹重叠检查（FP 是否形态相似的干扰）---")
pfp = collections.Counter((round(float(d["period"]), 2), round(float(d["dm"]), 1))
                          for u, d in ti.items() if d.get("label") == "pulsar")
same = sum(1 for i in fp if (round(float(ti[uids[i]]["period"]), 2),
                             round(float(ti[uids[i]]["dm"]), 1)) in pfp)
print("FP 中与某真实脉冲星 (period,dm) 相同的: %d / %d" % (same, len(fp)))
