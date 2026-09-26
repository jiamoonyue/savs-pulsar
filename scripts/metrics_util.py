"""不平衡检测指标 —— 全部委托 sklearn.metrics，不自实现。

项目规则（用户硬性要求）：有成熟库可复用的功能一律调用库，禁止重造轮子。
  confusion      -> sklearn.metrics.confusion_matrix
  pr_auc (AP)    -> sklearn.metrics.average_precision_score
  recall_at_fpr  -> sklearn.metrics.roc_curve
  pr_curve       -> sklearn.metrics.precision_recall_curve

口径注意：AP（PR 曲线下面积）与 ROC-AUC 不是一回事，不平衡数据下差异很大；
本模块主指标用 AP，不用 ROC-AUC。

用法: python scripts/metrics_util.py --selftest
"""
import argparse
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)


def confusion(finals, labels):
    """混淆矩阵。labels=[0,1] 固定顺序 → 1=正类=pulsar。"""
    m = confusion_matrix(np.asarray(labels, dtype=int),
                         np.asarray(finals, dtype=int), labels=[0, 1])
    tn, fp, fn, tp = m.ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def pr_auc(scores, labels):
    """Average Precision。注意 sklearn 参数序是 (y_true, y_score)。"""
    return float(average_precision_score(np.asarray(labels, dtype=int),
                                         np.asarray(scores, dtype=float)))


def recall_at_fpr(scores, labels, targets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)):
    """FPR ≤ t 时可达的最高召回 = roc_curve 上 fpr≤t 各点的 max(tpr)。"""
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=int),
                            np.asarray(scores, dtype=float))
    out = {}
    for t in targets:
        mask = fpr <= t
        out['%.4g' % t] = float(tpr[mask].max()) if mask.any() else 0.0
    return out


def pr_curve(scores, labels, max_points=101):
    """P–R 曲线采样点 [(recall, precision), ...]，按 recall 升序，供作图。"""
    prec, rec, _ = precision_recall_curve(np.asarray(labels, dtype=int),
                                          np.asarray(scores, dtype=float))
    if len(rec) == 0:
        return []
    order = np.argsort(rec)
    rec, prec = rec[order], prec[order]
    idx = np.linspace(0, len(rec) - 1, min(max_points, len(rec))).astype(int)
    return [(round(float(rec[i]), 5), round(float(prec[i]), 5)) for i in idx]


def _selftest():
    """解析真值自检（与实现无关，可证伪）。"""
    ok = True
    # 用例1：混淆矩阵
    c = confusion([1, 1, 0, 0], [1, 0, 1, 0])
    exp = {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    print('  [1] confusion =', c, '期望', exp)
    if c != exp:
        print('    FAIL'); ok = False
    # 用例2：完美排序 AP = 1.0
    ap = pr_auc([3, 2, 1, 0], [1, 1, 0, 0])
    print('  [2] 完美排序 AP = %.6f（期望 1.0）' % ap)
    if abs(ap - 1.0) > 1e-9:
        print('    FAIL'); ok = False
    # 用例3：反向（最差）排序。手算：降序标签序 [0,0,1,1]
    #   cum(tp,fp)=(0,1),(0,2),(1,2),(2,2) → 0.5×(1/3) + 0.5×(1/2) = 5/12
    ap2 = pr_auc([0, 1, 2, 3], [1, 1, 0, 0])
    print('  [3] 反向排序 AP = %.6f（手算 5/12=%.6f）' % (ap2, 5.0 / 12.0))
    if abs(ap2 - 5.0 / 12.0) > 1e-9:
        print('    FAIL'); ok = False
    # 用例4：recall@FPR —— 分数 3,2 为正类且排在前 → FPR=0 时召回 1.0
    raf = recall_at_fpr([3, 2, 1, 0], [1, 1, 0, 0], targets=(0.0, 0.5))
    print('  [4] recall@FPR =', raf, '期望 {"0":1.0,"0.5":1.0}')
    if abs(raf.get('0', -1) - 1.0) > 1e-9 or abs(raf.get('0.5', -1) - 1.0) > 1e-9:
        print('    FAIL'); ok = False
    # 用例5：P–R 曲线非空且 recall 单调不减
    pc = pr_curve([3, 2, 1, 0], [1, 1, 0, 0], max_points=5)
    recs = [p[0] for p in pc]
    mono = all(recs[i] <= recs[i + 1] for i in range(len(recs) - 1))
    print('  [5] pr_curve 点数=%d recall 单调不减=%s' % (len(pc), mono))
    if not pc or not mono:
        print('    FAIL'); ok = False
    # 用例6：随机大量样本，AP 与 ROC-AUC 应不同（口径差异哨兵）
    rng = np.random.RandomState(0)
    s = rng.rand(500); y = (rng.rand(500) < 0.2).astype(int)
    if y.sum() > 0 and y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        a, r = pr_auc(s, y), roc_auc_score(y, s)
        print('  [6] 随机数据 AP=%.4f  ROC-AUC=%.4f（两者口径不同，不应相等）' % (a, r))
        if abs(a - r) < 1e-6:
            print('    WARN: 两者相同，疑似调用错函数')
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()
