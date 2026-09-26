"""数据集B（易分类）构建：正样本=直接加载 region_gpps pulsar（4面板已裁），负样本=裁ACCEL。

输出 data/accel/{support,query}_k{k}_s{s}.jsonl：
  - 每样本 {uid, label, panels:{freq_phase,time_phase,dm_curve,profile}}
  - 正样本：复用 region_gpps 的 pulsar（panels 路径直接引用）
  - 负样本：从 ACCEL 图裁 4 面板
支持集/查询集：pulsar/rfi 各抽 k 支持，剩余做查询（rfi 子采样）。

用法: python scripts/build_accel_v2.py --k 25 --seed 0
"""
import argparse, json, os, random
from PIL import Image

COORD = json.load(open("/SAVS_DATA_ROOT/ApJS实验/裁剪坐标.json"))
PANEL_KEY = {"freq_phase": "③freq_phase", "time_phase": "②time_phase",
             "dm_curve": "④DM_curve", "profile": "①profile", "text_info": "⑤text_info"}
POS_SRC = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/data/region_gpps/all.jsonl"
POS_CROP = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops"
NEG_SRC = "/SAVS_DATA_ROOT/CCF-A/新FAST测试/负样本"
OUT_CROP = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops_accel"
OUT_DATA = "data/accel"


def crop_panels(img, uid):
    """负样本 ACCEL 裁 4 面板，返回 {panel: path}。"""
    paths = {}
    for pk, ck in PANEL_KEY.items():
        box = COORD["panels"][ck]["pixel"]
        crop = img.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
        os.makedirs(f"{OUT_CROP}/{uid}", exist_ok=True)
        p = f"{OUT_CROP}/{uid}/{pk}.png"
        crop.save(p)
        paths[pk] = p
    return paths


def split(samples, k, seed, rfi_query):
    """每类 k 支持（平衡）；查询=剩余pulsar + rfi子采样。"""
    rng = random.Random(seed)
    by = {"pulsar": [], "rfi": []}
    for s in samples:
        by[s["label"]].append(s)
    sup = []
    for lab in ("pulsar", "rfi"):
        pool = by[lab][:]; rng.shuffle(pool)
        sup.extend(pool[:k]); by[lab] = pool[k:]
    qry = by["pulsar"]
    rfipool = by["rfi"][:]; rng.shuffle(rfipool)
    qry += rfipool[:rfi_query]
    return sup, qry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--rfi_query", type=int, default=2000)
    args = ap.parse_args()

    # 正样本：直接加载 region_gpps pulsar（4面板复用），补 text_info（从 crops 取）
    pos = [json.loads(l) for l in open(POS_SRC)]
    pos = [x for x in pos if x["label"] == "pulsar"]
    for it in pos:
        # 正样本 text_info = crops 里已有的 text_info.png（同 uid）
        ti = f"{POS_CROP}/{it['uid']}/text_info.png"
        it["panels"]["text_info"] = ti
    print(f"正样本 pulsar 加载: {len(pos)} 张（4面板复用region_gpps + text_info从crops取）")

    # 负样本：ACCEL 裁 4 面板
    neg_files = sorted(os.listdir(NEG_SRC))
    neg = []
    for i, fn in enumerate(neg_files[:5000]):   # 先取5000控制量
        uid = f"a{i:06d}"
        img = Image.open(f"{NEG_SRC}/{fn}").convert("RGB")
        if img.size != (2200, 1700):
            img = img.resize((2200, 1700))
        panels = crop_panels(img, uid)
        neg.append({"uid": uid, "label": "rfi", "panels": panels})
    print(f"负样本 ACCEL 裁剪: {len(neg)} 张")

    samples = pos + neg
    os.makedirs(OUT_DATA, exist_ok=True)
    for seed in [int(x) for x in args.seeds.split(",")]:
        sup, qry = split(samples, args.k, seed, args.rfi_query)
        jsonl_s = f"{OUT_DATA}/support_k{args.k}_s{seed}.jsonl"
        jsonl_q = f"{OUT_DATA}/query_k{args.k}_s{seed}.jsonl"
        with open(jsonl_s, "w") as f:
            for s in sup: f.write(json.dumps(s) + "\n")
        with open(jsonl_q, "w") as f:
            for s in qry: f.write(json.dumps(s) + "\n")
        print(f"seed {seed}: support={len(sup)} (pul {sum(1 for s in sup if s['label']=='pulsar')}/rfi {sum(1 for s in sup if s['label']=='rfi')}) "
              f"query={len(qry)} (pul {sum(1 for s in qry if s['label']=='pulsar')}/rfi {sum(1 for s in qry if s['label']=='rfi')})")


if __name__ == "__main__":
    main()
