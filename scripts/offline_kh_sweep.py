"""离线 K/H 扫描：用全量头缓存秒级评估所有 (k, H, 选头准则)。

原理：查询集不随 k/seed 变 → 查询侧全量 784 头缓存（*_fullqryfeat.pt）对所有
      (k, seed, H) 通用；支持集缓存按 (k, seed) 各有（*_supfeat.pt）。
      → 选头 + 质心 + 投票全部可离线复算。

**口径保证**：直接 import run_regionaware 的 region_head_scoring / select_heads /
build_centroids / classify_sample_v2，不另写一套。
**自校验**：对已知配置（k=25, H=40, insample）应复现 RESULTS.md 的 F1；偏差大则报警。

用法:
  python scripts/offline_kh_sweep.py --data region --cache features_htru_full \
      --panels dm,phase_subband,phase_subintegration --ks 5,15,25,50 --seeds 0,1,2,3,4 \
      --heads 20,40,80,120,200
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import metrics_util as M

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
torch.set_grad_enabled(False)


def build(heads_list, dominant, sup, lab, q, mode, panels_n):
    """复刻 run_regionaware 的选头流程（用其本尊函数）。"""
    import run_regionaware as RA
    scores = RA.region_head_scoring(sup, torch.tensor(lab), mode=mode)
    h_idx, dom = RA.select_heads(scores, q=q, k=heads_list)
    cents = RA.build_centroids(sup, torch.tensor(lab), h_idx, dom)
    return h_idx, dom, cents


def vote_all(qf, h_idx, dom, cents, method='a1'):
    """逐样本投票 → 分数（票差）。qf: [M, H, 128]"""
    import run_regionaware as RA
    n = qf.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        votes, final, _ = RA.classify_sample_v2(qf[i], h_idx, dom, cents, None, method)
        s = int(sum(votes))
        out[i] = s - (len(votes) - s)
    return out


def f1_of(score, lab, thr=0):
    pred = (score > thr).astype(int)
    tp = int(((pred == 1) & (lab == 1)).sum()); fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='数据集标签（region / region_gpps / 新FAST测试）')
    ap.add_argument('--cache', required=True, help='缓存目录')
    ap.add_argument('--panels_all', required=True, help='该数据集的面板名（逗号分隔，需与缓存时的顺序一致）')
    ap.add_argument('--data_root', default=None, help='jsonl 所在根（默认 SAVs-main/data/<data>）')
    ap.add_argument('--ks', default='5,15,25,50')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--heads', default='40')
    ap.add_argument('--q', type=int, default=2)
    ap.add_argument('--mode', default='insample', choices=['insample', 'loo'])
    ap.add_argument('--prompt', default='empty')
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    Hs = [int(x) for x in args.heads.split(',')]
    # 面板顺序：与缓存时的 PANELS 一致（存的是主导面板索引）
    import run_regionaware as RA
    RA.PANELS = [p.strip() for p in args.panels_all.split(',')]
    root = args.data_root or os.path.join(HERE, '..', 'data', args.data)

    # 查询缓存（跨 k/seed 通用）：文件名里的 k/s 只是【提取时】的标签（查询集固定，
    # 与 k/seed 无关），故除按 ks/seeds 精确匹配外，再兜底 glob 任一份同数据集的 fullqryfeat。
    qry = None
    _cands = [os.path.join(args.cache, '%s_support_k%d_s%d_%s_fullqryfeat.pt'
                           % (args.data, kk, sd, args.prompt))
              for kk in ks for sd in seeds]
    _cands += sorted({(f.rsplit('.chunk', 1)[0] if '.chunk' in f else f)
                      for f in __import__('glob').glob(
                          os.path.join(args.cache, '%s_support_k*_s*_%s_fullqryfeat.pt*'
                                       % (args.data, args.prompt)))})
    for p in _cands:
        # CHUNKED_LOAD: 优先单文件；否则拼分块
        if os.path.exists(p):
            qry = torch.load(p, map_location='cpu')
            print('查询缓存: %s  %s' % (os.path.basename(p), tuple(qry.shape)))
            break
        _chunks = sorted(__import__('glob').glob(p + '.chunk*.pt'),
                         key=lambda x: int(x.rsplit('chunk', 1)[1].split('.')[0]))
        if _chunks:
            qry = torch.cat([torch.load(c, map_location='cpu') for c in _chunks], dim=0)
            print('查询缓存(分块 %d): %s  %s' % (len(_chunks), os.path.basename(p), tuple(qry.shape)))
            break
    if qry is None:
        print('!! 找不到 fullqryfeat 缓存（列目录）:')
        import glob
        for f in glob.glob(os.path.join(args.cache, '*fullqryfeat*')):
            print('   ', os.path.basename(f))
        sys.exit(2)
    # 保持 bf16 与管线一致（转 fp32 会翻转边界判断）
    ql = np.array([1 if json.loads(l)['label'] == 'pulsar' else 0
                   for l in open(os.path.join(root, 'query_k%d_s%d.jsonl' % (ks[0], seeds[0])))])

    print()
    print('%-6s %-6s %-7s %-17s %-17s %-17s %-17s %-17s'
          % ('k', 'H', 'n_seed', 'F1(mean±std)', 'P(mean±std)', 'R(mean±std)',
             'AP(mean±std)', 'r@1%FPR(mean±std)'))
    print('-' * 110)
    for k in ks:
        for H in Hs:
            res = []
            for s in seeds:
                sp = os.path.join(args.cache, '%s_support_k%d_s%d_%s_supfeat.pt' % (args.data, k, s, args.prompt))
                if not os.path.exists(sp):
                    continue
                sup = torch.load(sp, map_location='cpu').float()
                sl = [1 if json.loads(l)['label'] == 'pulsar' else 0
                      for l in open(os.path.join(root, 'support_k%d_s%d.jsonl' % (k, s)))]
                h_idx, dom, cents = build(H, None, sup, sl, args.q, args.mode, len(RA.PANELS))
                # 选头投影到查询（复刻 extract_query_features）
                qf = torch.stack([qry[:, int(dom[j]), int(h_idx[j])] for j in range(len(h_idx))], dim=1)
                sc = vote_all(qf, h_idx, dom, cents, 'a1')
                # AP / recall@1%FPR 用连续票差（margin），与管线口径一致
                res.append(f1_of(sc, ql) + (M.pr_auc(sc, ql),
                                            M.recall_at_fpr(sc, ql)['0.01']))
            if not res:
                print('%-6d %-6d %-7s %s' % (k, H, '0', '（无支持集缓存）'))
                continue
            f1s = np.array([r[0] for r in res])
            ps = np.array([r[1] for r in res])
            rs = np.array([r[2] for r in res])
            aps = np.array([r[3] for r in res])
            rafs = np.array([r[4] for r in res])
            print('%-6d %-6d %-7d %-17s %-17s %-17s %-17s %-17s'
                  % (k, H, len(res),
                     '%.4f±%.4f' % (f1s.mean(), f1s.std()),
                     '%.4f±%.4f' % (ps.mean(), ps.std()),
                     '%.4f±%.4f' % (rs.mean(), rs.std()),
                     '%.4f±%.4f' % (aps.mean(), aps.std()),
                     '%.4f±%.4f' % (rafs.mean(), rafs.std())))


if __name__ == '__main__':
    main()
