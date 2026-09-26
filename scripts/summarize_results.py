"""汇总 results/ 为 results/RESULTS.md —— 论文唯一数据文件。

结构（按数据集组织，主表在前、明细后置）：
  §0 摘要（全数据集 headline）
  §1..N 逐数据集：主结果表 → 消融表(若有) → 逐 seed 明细
  附录 A 配置规格 / B 诊断 / C 复现 / D 数据资产

⚠ 管线数字与离线复算数字分节呈现，各自标注来源。

用法: python scripts/summarize_results.py [--heads 40]
"""
import argparse
import datetime
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(RES, 'RESULTS.md')
R1 = '0.01'
BACKUP = '_archive/FAST_results_final_20260911'

# 数据集：展示名 + 所属数据集分组 + 说明
DS = {
    'pulsar_gpps': dict(name='整图（全局 SAVs）', group='FAST-Hard',
                        desc='FAST 巡天 GPPS 候选体，形态高度难分。基线对照臂。'),
    'region_gpps': dict(name='四图（区域感知）', group='FAST-Hard',
                        desc='同数据集的区域感知臂。与上一张表严格配对（同支持集/同查询集/同 seed）。'),
    'region': dict(name='四图（区域感知）', group='HTRU1（中集，Parkes 巡天）',
                   desc='HTRU Medlat，3 面板（dm / phase-subband / phase-subintegration），正负比 1:49。'),
    'whole': dict(name='整图（全局 SAVs）', group='HTRU1（中集，Parkes 巡天）', desc=''),
    '新FAST测试': dict(name='四图（区域感知）', group='FAST-Easy', desc='FAST easy set: morphologically separable candidates.'
                   '⚠ **本表数值由全量头缓存离线复算（CPU）得到，与管线（GPU）存在 '
                   '±0.03 量级随机差、符号不定，不是管线实跑值**；管线仅跑过 6/20 个 part。'),
    'MiraBest数据集': dict(name='整图（全局 SAVs）', group='MiraBest（对比集）', desc='FR-I vs FR-II 形态分类（339/431）。**每源仅一张射电图，无面板分解** → 只能跑整图臂。类别映射：fri→pulsar, frii→rfi（与管线口径一致）。'),
}
ORDER = ['FAST-Hard', 'HTRU1（中集，Parkes 巡天）', 'FAST-Easy', 'MiraBest（对比集）']
# 主表数值来源为【离线复算】而非管线实跑的数据集（§0 摘要里加 dagger 标）
OFFLINE_MAIN = {'新FAST测试'}
OFFLINE_DESC = {
    'fast_kh_sweep': ('FAST-Hard · K×H 消融（离线复算）',
                      '用查询集全量 784 头缓存复算（零 GPU）。'
                      '⚠ **口径（实测）**：离线（CPU）与管线（GPU）**逐 seed 均不同**，'
                      'delta 符号不一致、|max| 约 0.03（ACCEL k5/H40 配对实测：mean -0.0127、'
                      'std 0.0177、|max| 0.0315，0/5 seed 逐位相同）'
                      '→ **本表只用于比较 k 与 H 的相对趋势，绝对值以管线表为准**。'),
    'htru_kh_sweep': ('HTRU1 · 头数 H 消融（离线复算）',
                      '用**查询集全量 784 头缓存**复算（零 GPU）。'
                      '⚠ **口径（实测）**：离线（CPU）与管线（GPU）**逐 seed 均不同**，'
                      'delta 符号不一致、|max| 约 0.03（同族实测见 FAST 表注）'
                      '→ **本表用于比较 H 与 k 的相对趋势，绝对数字以管线表为准**。'),
    'accel_kh_sweep': ('FAST-Easy · K×H 消融（离线复算）',
                       '同 HTRU1 方式：全量头缓存离线复算。'
                       '⚠ **口径（实测，本数据集上量的）**：k5/H40 配对 5 个 seed，'
                       'delta = -0.007 / **+0.016** / -0.031 / -0.031 / -0.010（**符号不一致**），'
                       'mean -0.0127、std 0.0177 → 随机扰动而非恒定偏移，**绝对值以管线为准**。'),
    'mira_kh_sweep': ('MiraBest（对比集）· 整图 K×H 消融（离线复算）',
                      '整图臂全量 784 头缓存离线复算（零 GPU，复用 run_baseline 的选头口径）。'
                      '⚠ 整图路径与管线**逐位等价**（实测 max abs diff = 0，**不是 +0.005 口径差**）：'
                      'k=50/H=40 复算 0.6821 与管线 RESULTS.md 的 0.6821 完全一致。'),
}


def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def _ms(v):
    a = np.array([x for x in v if x is not None], dtype=float)
    return (None, None) if len(a) == 0 else (float(a.mean()), float(a.std()))


def agg(d):
    if not d:
        return None
    f1 = np.array([v['f1'] for v in d.values()], dtype=float)
    ap_m, ap_s = _ms([v.get('pr_auc') for v in d.values()])
    r1_m, r1_s = _ms([v.get('recall_at_fpr', {}).get(R1) for v in d.values()])
    p = np.array([v['precision'] for v in d.values()], dtype=float)
    r = np.array([v['recall'] for v in d.values()], dtype=float)
    return dict(n=len(d), f1=float(f1.mean()), f1s=float(f1.std()),
                p=float(p.mean()), ps=float(p.std()),
                r=float(r.mean()), rs=float(r.std()),
                ap=ap_m, aps=ap_s, r1=r1_m, r1s=r1_s)


def fmt(v, s=None):
    if v is None:
        return '–'
    return ('%.4f ± %.4f' % (v, s)) if s is not None else ('%.4f' % v)


def discover():
    g = {}
    for f in sorted(os.listdir(RES)):
        if not f.endswith('.json'):
            continue
        m = re.match(r'(.+?)_k(\d+)_h(\d+)_', f)
        if not m:
            continue
        ds, k, h = m.group(1), int(m.group(2)), int(m.group(3))
        if ds == 'baseline':
            ds = 'pulsar_gpps'
        g.setdefault((ds, h), []).append((k, os.path.join(RES, f)))
    return g


def parse_offline():
    out = []
    _want = list(OFFLINE_DESC.keys())
    _files = glob.glob(os.path.join(ROOT, 'logs', '*kh_sweep*.log'))
    _files.sort(key=lambda f: _want.index(os.path.basename(f)[:-4])
                if os.path.basename(f)[:-4] in _want else len(_want))
    for lf in _files:
        rows = []
        for ln in open(lf):
            p = ln.split()
            if len(p) >= 6 and p[0].isdigit() and p[1].isdigit():
                rows.append(p[:6])
        if rows:
            key = os.path.basename(lf).replace('.log', '')
            out.append((key, rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', type=int, default=40)
    args = ap.parse_args()
    H = args.heads

    byds = {}
    for (ds, h), items in discover().items():
        if h != H:
            continue
        ks = sorted(set(k for k, _ in items))
        merged = {}
        for k, f in items:
            d = load(f)
            if d and (k not in merged or len(d) > len(merged[k][0])):
                merged[k] = (d, f)
        if merged:
            byds[ds] = dict(ks=ks, data={k: (d, agg(d), f) for k, (d, f) in merged.items()})

    # 按 group 归并
    groups = {}
    for ds, info in byds.items():
        gname = DS.get(ds, {}).get('group', ds)
        groups.setdefault(gname, []).append(ds)

    L = []
    A = L.append
    A('# SAVs 脉冲星候选体分类 · 实验结果\n')
    A('> 自动生成：%s ｜ 生成器 `scripts/summarize_results.py`'
      % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    A('> **请勿手改**（重跑覆盖）；要改内容请改生成器。')
    A('> 备份：`%s/`（FAST 全部结果 + part）\n' % BACKUP)
    A('指标：F1 / P / R（单点，阈值 0）；AP（PR-AUC）与 recall@1%FPR（不平衡稳健）。')
    A('每格为 **5 个 seed 的 mean ± std**；**★ = 该表最优**。\n')
    A('**k = 每类支持样本数**（few-shot 标注量）：k=25 → 25 正 + 25 负 = **50 个标注**。')
    A('**查询集固定**，不随 k / seed 变化（同一份考卷，只换复习材料）。')
    A('因此 seed 之间的差异**只反映「标注了哪几张图」**，不含查询样本抽样的噪声。\n')

    # ---------- §0 摘要 ----------
    A('## 0. 摘要\n')
    A('| 数据集 | 臂 | 最优 k | F1 (mean±std) | P (mean±std) | R (mean±std) | AP (mean±std) |')
    A('|---|---|---|---|---|---|---|')
    best_rows = []
    for gname in ORDER:
        if gname not in groups:
            continue
        for ds in groups[gname]:
            info = byds[ds]
            best = None
            for k in info['ks']:
                a = info['data'][k][1]
                if a and (best is None or a['f1'] > best[1]['f1']):
                    best = (k, a)
            if not best:
                continue
            k, a = best
            _mk = ' †' if ds in OFFLINE_MAIN else ''
            A('| %s | %s | %d | **%s** ★%s | %s | %s | %s |'
              % (gname, DS.get(ds, {}).get('name', ds), k, fmt(a['f1'], a['f1s']), _mk,
                 fmt(a['p'], a.get('ps')), fmt(a['r'], a.get('rs')), fmt(a['ap'], a.get('aps'))))
            best_rows.append((gname, ds, k, a))
    if any(r[1] in OFFLINE_MAIN for r in best_rows):
        A('> † = 该行数值由**离线复算**得到（全量头缓存 + CPU），与管线（GPU）存在 '
          '±0.03 量级随机差、符号不定，**不是管线实跑值**。详见对应数据集小节。' + chr(10))
    A('')

    # ---------- 逐数据集 ----------
    sec = 0
    for gname in ORDER:
        if gname not in groups:
            continue
        sec += 1
        A('## %d. %s\n' % (sec, gname))
        for ds in groups[gname]:
            meta = DS.get(ds, {})
            A('### %s\n' % meta.get('name', ds))
            if meta.get('desc'):
                A(meta['desc'] + '\n')
            _ks_here = byds[ds]['ks']  # KNOTE_V2
            if len(_ks_here) <= 2:
                A('> 本数据集**管线实跑**的 k = %s；其余 k 与头数 H 的扫描见下方「消融实验」节（离线复算，与管线存在 ±0.03 量级随机差（符号不定），**仅用于趋势比较**）。\n'
                  % '/'.join(str(x) for x in _ks_here))
            A('| k | n_seed | F1 (mean±std) | P (mean±std) | R (mean±std) | AP (mean±std) | recall@1%FPR (mean±std) |')
            A('|---|---|---|---|---|---|---|')
            _rows = [(k, byds[ds]['data'][k]) for k in byds[ds]['ks'] if byds[ds]['data'][k][1]]
            _bestk = max(_rows, key=lambda t: t[1][1]['f1'])[0] if _rows else None  # BESTMARK_V1
            for k, (d, a, f) in _rows:
                _mark = ' ★' if k == _bestk else ''
                A('| %d | %d | **%s**%s | %s | %s | %s | %s |'
                  % (k, a['n'], fmt(a['f1'], a['f1s']), _mark,
                     fmt(a['p'], a.get('ps')), fmt(a['r'], a.get('rs')),
                     fmt(a['ap'], a.get('aps')), fmt(a['r1'], a.get('r1s'))))
            if _bestk is not None:
                A('')
                A('> ★ = 本表最优 F1（k=%d）。' % _bestk)
            A('')

    # ---------- 离线消融 ----------
    offs = parse_offline()
    if offs:
        sec += 1
        A('## %d. 消融实验\n' % sec)
        for key, rows in offs:
            title, note = OFFLINE_DESC.get(key, (key, '离线复算结果。'))
            A('### %s\n' % title)
            A(note + '\n')
            A('| k | H | n_seed | F1 (mean±std) | P | R |')
            A('|---|---|---|---|---|---|')

            def _f1v(r):  # BESTMARK_V1: 从 '0.9105±0.0091' 取数值
                try:
                    return float(r[3].split('±')[0])
                except Exception:
                    return -1.0
            _b = max(rows, key=_f1v) if rows else None
            for r in rows:
                _m = ' ★' if (r == _b) else ''
                A('| %s | %s | %s | **%s**%s | %s | %s |' % (r[0], r[1], r[2], r[3], _m, r[4], r[5]))
            if _b:
                A('')
                A('> ★ = 本表最优 F1（k=%s, H=%s）。' % (_b[0], _b[1]))
            A('')

    # ---------- 附录 ----------
    A('## 附录 A · 配置规格\n')
    A('| 项 | 值 |')
    A('|---|---|')
    A('| 模型 | Qwen2.5-VL-7B-Instruct（bf16, sdpa, **冻结**，无梯度） |')
    A('| 头数 H | %d（每样本取 top-H 注意力头） |' % H)
    A('| 支持集 | 每类 k 张；**正负两类都随 seed 重抽** |')
    A('| 查询集 | **固定**（不随 k、不随 seed 变） |')
    A('| 区域感知 | 4 面板联合 + 区域评分 + Q=2 覆盖约束 + 分组投票 + conf_cross |')
    A('| prompt | 空串（未使用任务语义文本） |')
    A('| 指标库 | `sklearn.metrics`（见 `DEV_PROTOCOL.md` §0：禁止自实现） |')
    A('')
    A('## 附录 B · 关键诊断（设计依据，均为实测）\n')
    A('| 结论 | 证据 |')
    A('|---|---|')
    A('| 假阳"自信地错"，阈值/校准类改进**上界≈0** | FAST k=25 的 FP 的 \\|margin\\| 中位 22/40；oracle 最优阈值 F1=0.7602 < 当前 0.7615 |')
    A('| 聚合类改进**无效** | 判据用 cosine（已 L2 不变）：SimpleShot Δ=−0.0013；连续分数 oracle 上界 +0.0008 |')
    A('| 转导质心细化**无效** | Δ=−0.0385 |')
    A('| **约一半假阳随支持集变** | 跨 seed FP 集合 Jaccard=0.517 → 选头/支持集方向仍有空间 |')
    A('| 选头准则有 **in-sample 偏差** | Δ=+0.071；top-40 选头跨 seed 仅重叠 15–22/40 |')
    A('')
    A('## 附录 C · 复现信息\n')
    A('- 划分：`scripts/gpps_data.py::split`（查询集固定 `query_seed=42`）')
    A('- 驱动：`run_l0_sweep.py` / `run_region_sweep.py`（`--data_dir` / `--panels` 支持多数据集）')
    A('- 缓存：`--cache_dir` 存支持集全量 784 头；查询特征跨 k/seed 复用（`--support_only`）')
    A('- 合并/汇总：`merge_l0.py` / `merge_region.py` / `summarize_results.py`')
    A('- 指标自检：`python scripts/metrics_util.py --selftest`')
    A('')
    A('## 附录 D · 数据资产\n')
    A('- merged 结果：`results/*_k*_h%d_*.json`' % H)
    A('- part：`results/l0_parts/`、`results/region_parts/`')
    A('- 特征缓存：`features/`（FAST）、`features_htru_full/`（HTRU1）、`features_accel/`、`features_mira/`')
    A('- **备份**：`%s/`' % BACKUP)

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print('wrote %s (%d 行)' % (OUT, len(L)))
    for gname, ds, k, a in best_rows:
        print('  %-26s %-14s k=%-3d F1=%.4f' % (gname, DS.get(ds, {}).get('name', ds), k, a['f1']))


if __name__ == '__main__':
    main()
