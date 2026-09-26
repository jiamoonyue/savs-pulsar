"""从离线 K×H 扫描日志中，为每个数据集选出最优 (k, H)。

输出 results/best_kh.json：
  {"<dataset>": {"k":..., "H":..., "f1":..., "f1_std":..., "src": "<log>"}, ...}

⚠ 结论用于「搜索」最优配置；论文主数字须用管线实跑该 (k,H) 确认
  （离线复算比管线高约 0.005）。

用法:
  python scripts/pick_best_kh.py [--selftest]
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
LOGS = os.path.join(ROOT, 'logs')
OUT = os.path.join(ROOT, 'results', 'best_kh.json')


def parse_log(path):
    """返回 [(k, H, n_seed, f1, f1_std), ...]"""
    rows = []
    for ln in open(path):
        p = ln.split()
        if len(p) < 4 or not p[0].isdigit() or not p[1].isdigit():
            continue
        m = re.match(r'([\d.]+)(?:±([\d.]+))?', p[3])
        if not m:
            continue
        f1 = float(m.group(1))
        sd = float(m.group(2)) if m.group(2) else 0.0
        rows.append((int(p[0]), int(p[1]), int(p[2]), f1, sd))
    return rows


def best_of(rows):
    return max(rows, key=lambda r: r[3]) if rows else None


def _selftest():
    """自检：构造已知排序的假日志，验证选出的是最高 F1（而非最后一个或第一个）。"""
    import tempfile
    a = """    5      20     5       0.8032±0.0965     0.7893     0.8541
15     40     5       0.9192±0.0223     0.9325     0.9073
25     40     5       0.9161±0.0096     0.9159     0.9172
50     20     5       0.9243±0.0091     0.9318     0.9174
50     40     5       0.9224±0.0088     0.9267     0.9185
"""
    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as f:
        f.write(a)
        p = f.name
    rows = parse_log(p)
    os.unlink(p)
    b = best_of(rows)
    ok = True
    print('  解析出 %d 行' % len(rows))
    if len(rows) != 5:
        print('    FAIL: 行数应为 5，实得 %d' % len(rows)); ok = False
    if b is None or (b[0], b[1]) != (50, 20):
        print('    FAIL: 最优应为 (50,20)，实得 %s' % (b,)); ok = False
    else:
        print('  最优 = k=%d H=%d F1=%.4f  ✓' % (b[0], b[1], b[3]))
    # 边界：空日志
    if best_of([]) is not None:
        print('    FAIL: 空输入应返回 None'); ok = False
    # 边界：并列时取先出现者（max 语义）
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)

    out = {}
    for lf in sorted(glob.glob(os.path.join(LOGS, '*kh_sweep*.log'))):
        name = os.path.basename(lf).replace('.log', '')
        rows = parse_log(lf)
        b = best_of(rows)
        if b is None:
            print('  %-20s 无有效行' % name); continue
        ds = name.replace('_kh_sweep', '')
        out[ds] = {'k': b[0], 'H': b[1], 'f1': b[3], 'f1_std': b[4],
                   'n_seed': b[2], 'n_cfg_scanned': len(rows), 'src': os.path.basename(lf)}
        print('  %-12s 扫了 %2d 组 → 最优 k=%-3d H=%-4d F1=%.4f' % (ds, len(rows), b[0], b[1], b[3]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print('已写 %s' % OUT)
    print('⚠ 这些是【离线搜索】结果；论文主数字须用管线实跑该 (k,H) 确认。')


if __name__ == '__main__':
    main()
