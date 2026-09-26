# -*- coding: utf-8 -*-
"""held-out 流端到端指标（阶段4 收尾脚本；4 个分块齐全后运行）。

输出审稿人第 1 条要求的全部量：
  输入候选体总数 / 已知脉冲星数 / 默认阈值下保留数 / 脉冲星召回(来自标注评测集) /
  非脉冲星剔除比例 / 90%、95%、99% 召回下的保留数与剔除比例。
"""
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


def f1_pr(score, lab):
    pred = (score > 0).astype(int)
    tp = int(((pred == 1) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r


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


# ---- 部署配置：seed 0 的 k=50 支持集（固定，不在 held-out 上做任何调整）----
sup = torch.load("%s/region_gpps_support_k%d_s%d_empty_supfeat.pt" % (CACHE, K, S),
                 map_location="cpu").float()
lab_s = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                  for l in open("%s/support_k%d_s%d.jsonl" % (DATA, K, S))])
scores = RA.region_head_scoring(sup, torch.tensor(lab_s), mode="insample")
h_idx, dom = RA.select_heads(scores, q=Q, k=H)
cents = RA.build_centroids(sup, torch.tensor(lab_s), h_idx, dom)

# ---- held-out 流 ----
chunks = sorted(glob.glob("./heldout_chunk*.pt"),
                key=lambda x: int(x.rsplit("chunk", 1)[1].split(".")[0]))
assert len(chunks) == 4, "分块不齐: %s" % chunks
stream = json.load(open("./heldout_stream.json"))
hf = torch.cat([torch.load(c, map_location="cpu").float() for c in chunks]).float()
print("held-out 流: %d 候选体  缓存 %s" % (hf.shape[0], tuple(hf.shape)))
hlab = np.array([1 if r["label"] == "pulsar" else 0 for r in stream])
n_pul = int(hlab.sum())
hmargin = margins(hf, h_idx, dom, cents)
print("计算完成，开始出表……", flush=True)

retained = hmargin > 0
n_ret = int(retained.sum())
n_rej = len(hmargin) - n_ret
print()
print("=== 默认工作点（阈值 0）===")
print("输入候选体: %d   其中已知脉冲星: %d" % (len(hmargin), n_pul))
print("保留: %d (%.1f%%)   自动剔除: %d (%.1f%%)" % (n_ret, 100 * n_ret / len(hmargin),
                                                    n_rej, 100 * n_rej / len(hmargin)))
if n_pul:
    print("流内已知脉冲星召回: %d/%d" % (int(((retained) & (hlab == 1)).sum()), n_pul))

# ---- 召回-保留曲线（在标注评测集上测召回，同一分类器同一票差定义）----
qry = torch.load("%s/region_gpps_support_k25_s0_empty_fullqryfeat.pt" % CACHE,
                 map_location="cpu").float()
qlab = np.array([1 if json.loads(l)["label"] == "pulsar" else 0
                 for l in open("%s/query_k%d_s%d.jsonl" % (DATA, K, S))])
qmargin = margins(qry, h_idx, dom, cents)
f1, p, r = f1_pr(qmargin, qlab)
print("\n标注评测集自校验（同分类器）: F1=%.4f P=%.4f R=%.4f" % (f1, p, r))

order = np.argsort(-qmargin)
sorted_lab = qlab[order]
tp_cum = np.cumsum(sorted_lab)
recall_curve = tp_cum / qlab.sum()

print("\n=== 90% / 95% / 99% 脉冲星召回下的保留与剔除 ===")
print("%-10s %-12s %-14s %-16s %-16s" % ("召回", "阈值(票差)", "流内保留", "剔除比例", "评测集精确率"))
for target in (0.90, 0.95, 0.99):
    idx = int(np.searchsorted(recall_curve, target))
    idx = min(idx, len(recall_curve) - 1)
    thr = qmargin[order][idx]
    rec = recall_curve[idx]
    keep = int((hmargin >= thr).sum())
    prec = (recall_curve[idx] * qlab.sum()) / max((hmargin >= thr).sum() * 0 + idx + 1, 1)
    print("%-10.0f%% %-12d %-14d %-15.1f%% (评测集 P=%.3f)"
          % (100 * target, thr, keep, 100 * (1 - keep / len(hmargin)), prec))

print("\n完成。")
