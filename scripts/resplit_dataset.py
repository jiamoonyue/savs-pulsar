"""对已有 all.jsonl（{uid,label,panels}）按统一 split 重新生成支持/查询集。

用于 B(新FAST测试/panels_all.jsonl) 与 Parkes(data/parkes/all.jsonl)——
与 A(GPPS) 采用同一划分协议（固定查询 + 两类重抽 + 多 seed），
且整图/四图共用同一份 jsonl，天然逐样本对齐。

用法:
  python scripts/resplit_dataset.py --all <all.jsonl> --out <dir> --ks 5,15,25,50 --seeds 0,1,2,3,4
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from gpps_data import split  # noqa: E402


def write_jsonl(path, items):
    with open(path, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + chr(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', required=True, help='源 all.jsonl（每行 {uid,label,panels}）')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--ks', default='5,15,25,50')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--rfi_query', type=int, default=2000)
    ap.add_argument('--query_seed', type=int, default=42)
    ap.add_argument('--reserve', type=int, default=100)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.all)]
    print('samples=' + str(len(samples)) + ' ' + str(dict(Counter(s['label'] for s in samples))))
    os.makedirs(args.out, exist_ok=True)
    for k in [int(x) for x in args.ks.split(',')]:
        for seed in [int(x) for x in args.seeds.split(',')]:
            sup, qry = split(samples, k, seed, args.rfi_query, args.query_seed, args.reserve)
            write_jsonl(os.path.join(args.out, 'support_k' + str(k) + '_s' + str(seed) + '.jsonl'), sup)
            write_jsonl(os.path.join(args.out, 'query_k' + str(k) + '_s' + str(seed) + '.jsonl'), qry)
        n_pul = sum(1 for s in qry if s['label'] == 'pulsar')
        print('  k=' + str(k) + ': support=' + str(len(sup)) + ' query=' + str(len(qry)) + ' (pul ' + str(n_pul) + '/rfi ' + str(len(qry) - n_pul) + ')')
    print('DONE')


if __name__ == '__main__':
    main()
