# -*- coding: utf-8 -*-
"""导出保留率-召回率曲线数据：同一部署分类器（seed0, k=50, N=40, Q=2），
召回在标注评测集上测，保留在 held-out 流上测。输出 CSV。"""
import csv
import glob
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
K, H, Q, S = 50, 40, 2, 0


def margins(qry, h_idx, dom, cents):
    N = qry.shape[0]
    out = np.zeros(N)
    for i in range(N):
        v = []
        for j, h in enumerate(h_idx.tolist()):
            c0, c1 = cents[h]
            x = qry[i, int(dom[j]), h]
            s0 = torch.nn.functional.cosine_similarity(x.unsqueeze(0), c0.unsqueeze(0))
            s1 = torch.nn.functional.cosine_similarity(x.unsqueeze(0), c1.unsqueeze(0))
            v.append(1 if s1 > s0 else 0)
        out[i] = 2 * sum(v) - len(v)
    return out


sup = torch.load("%s/region_gpps_support_k%d_s%d_empty_supfeat.pt" % (CACHE, K, S),
                 map_location="cpu").float()
lab_s = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                  for l in open("%s/support_k%d_s%d.jsonl" % (DATA, K, S))])
scores = RA.region_head_scoring(sup, torch.tensor(lab_s), mode="insample")
h_idx, dom = RA.select_heads(scores, q=Q, k=H)
cents = RA.build_centroids(sup, torch.tensor(lab_s), h_idx, dom)

qry = torch.load("%s/region_gpps_support_k25_s0_empty_fullqryfeat.pt" % CACHE,
                 map_location="cpu").float()
qlab = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                 for l in open("%s/query_k%d_s%d.jsonl" % (DATA, K, S))])
qm = margins(qry, h_idx, dom, cents)

stream = json.load(open("./heldout_stream.json"))
chunks = sorted(glob.glob("./heldout_chunk*.pt"),
                key=lambda x: int(x.rsplit("chunk", 1)[1].split(".")[0]))
hf = torch.cat([torch.load(c, map_location="cpu").float() for c in chunks]).float()
hm = margins(hf, h_idx, dom, cents)
print("query N=%d (pulsar %d)  stream N=%d" % (len(qm), int(qlab.sum()), len(hm)), flush=True)

n_pos = int(qlab.sum())
ths = np.unique(np.concatenate([qm, [qm.min() - 1]]))[::-1]
rows = []
for t in ths:
    keep_q = qm >= t
    tp = int(((qm >= t) & (qlab == 1)).sum())
    rec = tp / n_pos
    ret = float((hm >= t).mean())
    rows.append((float(t), rec, ret))
rows.append((float(ths[-1] - 1), 1.0, 1.0))

# 只保留 (recall, retention) 发生变化的折点
dedup = []
for t, rec, ret in rows:
    if not dedup or (rec, ret) != (dedup[-1][1], dedup[-1][2]):
        dedup.append((t, rec, ret))

with open("./stream_curve.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "pulsar_recall", "stream_retained_frac"])
    for t, rec, ret in dedup:
        w.writerow([t, round(rec, 6), round(ret, 6)])
print("曲线折点数:", len(dedup))

# 关键工作点
print("\n%-12s %-14s %-16s" % ("召回", "阈值(票差)", "流内保留比例"))
for tgt in (0.7567, 0.90, 0.95, 0.99):
    best = None
    for t, rec, ret in dedup:
        if rec >= tgt:
            best = (t, rec, ret)
            break
    if best:
        print("%-12.4f %-14d %-16.4f" % (best[1], best[0], best[2]))
# 默认点
t0 = 0.0
tp0 = int(((qm >= t0) & (qlab == 1)).sum())
print("默认阈值0: 召回=%.4f 流内保留=%.4f" % (tp0 / n_pos, float((hm >= t0).mean())))
