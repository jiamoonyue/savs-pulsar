# -*- coding: utf-8 -*-
"""FAST hard (region_gpps)：用 (period, dm) 作为源指纹，审计 support/query 是否源级泄漏。"""
import collections
import json
import os

D = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/data/region_gpps/"
TI = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/data/textinfo/textinfo_all_k25.json"

if not os.path.exists(TI):
    cand = []
    for base, _, files in os.walk("/SAVS_DATA_ROOT/CCF-A/SAVs-main/data"):
        for f in files:
            if "textinfo" in f:
                cand.append(os.path.join(base, f))
    print("textinfo 候选:", cand)
    if cand:
        TI = cand[0]
    else:
        raise SystemExit("找不到 textinfo")

ti = json.load(open(TI))
print("textinfo 文件:", TI, " 条数:", len(ti))
k = list(ti)[0]
print("字段示例:", k, ti[k])

qry = [json.loads(l)["uid"] for l in open(D + "query_k50_s0.jsonl") if l.strip()]
sup = {}
for s in range(5):
    sup[s] = set(json.loads(l)["uid"] for l in open(D + "support_k50_s%d.jsonl" % s) if l.strip())

print("查询集候选体数:", len(qry), " (唯一 %d)" % len(set(qry)))
for s in range(5):
    print("  support 种子 %d: %d 行" % (s, len(sup[s])))


def fp(u):
    d = ti.get(u)
    if not d:
        return None
    return (round(float(d["period"]), 4), round(float(d["dm"]), 3))


pul = [u for u in ti if ti[u].get("label") == "pulsar"]
print("textinfo 中 pulsar 数:", len(pul), " rfi 数:", sum(1 for u in ti if ti[u].get("label") != "pulsar"))

groups = collections.defaultdict(list)
for u in pul:
    groups[fp(u)].append(u)
multi = {f: v for f, v in groups.items() if len(v) > 1}
print("同源(period+dm 相同)且含多个候选体的源数:", len(multi),
      " 涉及候选体:", sum(len(v) for v in multi.values()))
print("  每源候选体数分布:", dict(sorted(collections.Counter(len(v) for v in multi.values()).items())))

qry_fp = set(fp(u) for u in qry if u in ti)
print("查询集覆盖到的源指纹数:", len(qry_fp))

total_leak = 0
for s in range(5):
    sfp = set(fp(u) for u in sup[s] if u in ti)
    common = sfp & qry_fp
    total_leak += len(common)
    flag = "  <<< 源级泄漏" if common else ""
    print("种子 %d：support 源指纹 %d，与 query 交集 = %d%s" % (s, len(sfp), len(common), flag))
    for f in list(common)[:3]:
        print("    指纹 %s : %s" % (f, groups.get(f)))

print("五种子合计源级泄漏:", total_leak)

# 附：support 在种子之间的重叠（脉动星类可用池大小问题）
allp = set(u for u in pul)
print("\n--- 种间重叠（pulsar 类）---")
for a in range(5):
    for b in range(a + 1, 5):
        ov = len(sup[a] & sup[b])
        if ov:
            print("  种子%d ∩ 种子%d = %d" % (a, b, ov))
print("pulsar support 每种子 50 个；全体 pulsar 只有 %d 个" % len(allp))
