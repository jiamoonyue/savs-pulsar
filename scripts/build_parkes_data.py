"""构建 Parkes 数据集 jsonl：读整图(pulsars/nonpulsars)，用 FAST 坐标裁 5 面板。

Parkes 布局与 FAST 相同（PRESTO 4 面板 + text_info），坐标通用。
子采样以控制规模；支持集/查询集划分（同 FAST 协议：rfi 查询子采样）。
输出 data/parkes/{support,query}_k{k}_s{s}.jsonl。

用法: python scripts/build_parkes_data.py --k 25 --seed 0 --n_pulsar 800 --n_rfi 800
"""
import argparse, json, os, random
from PIL import Image

COORD = json.load(open("/SAVS_DATA_ROOT/CCF-A/裁剪坐标.json"))
BASE = "/SAVS_DATA_ROOT/CCF-A/datasets/Parkes/sgan_lowlat_dataset_png"
OUT = "data/parkes"
OUT_CROP = "data/parkes/crops"
PANEL_KEY = {"freq_phase": "③freq_phase", "time_phase": "②time_phase",
             "dm_curve": "④DM_curve", "profile": "①profile", "text_info": "⑤text_info"}


def crop5(img, uid):
    paths = {}
    os.makedirs(f"{OUT_CROP}/{uid}", exist_ok=True)
    for pk, ck in PANEL_KEY.items():
        box = COORD["panels"][ck]["pixel"]
        crop = img.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
        p = f"{OUT_CROP}/{uid}/{pk}.png"
        crop.save(p)
        paths[pk] = p
    return paths


def build_all(n_pulsar, n_rfi, seed):
    rng = random.Random(seed)
    samples = []
    for label, dirname, n in [("pulsar", "pulsars", n_pulsar), ("rfi", "nonpulsars", n_rfi)]:
        files = sorted(os.listdir(f"{BASE}/{dirname}"))
        rng.shuffle(files); files = files[:n]
        for fn in files:
            uid_dir = f"{OUT_CROP}/{fn}"
            if os.path.exists(f"{uid_dir}/freq_phase.png"):
                # 已裁过，复用
                panels = {p: f"{uid_dir}/{p}.png" for p in PANEL_KEY}
                samples.append({"uid": fn, "label": label, "panels": panels})
            else:
                img = Image.open(f"{BASE}/{dirname}/{fn}").convert("RGB")
                if img.size != (2200, 1700):
                    img = img.resize((2200, 1700))
                samples.append({"uid": fn, "label": label, "panels": crop5(img, fn)})
    return samples


from gpps_data import split  # noqa: E402  统一划分入口（见 gpps_data.py）




def write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--n_pulsar", type=int, default=800)
    ap.add_argument("--n_rfi", type=int, default=800)
    ap.add_argument("--rfi_query", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    samples = build_all(args.n_pulsar, args.n_rfi, seed=7)
    print(f"Parkes 裁剪(+坐标): {len(samples)} (pulsar {sum(1 for s in samples if s['label']=='pulsar')}/rfi {sum(1 for s in samples if s['label']=='rfi')})")
    write_jsonl(f"{OUT}/all.jsonl", samples)
    for seed in [int(x) for x in args.seeds.split(",")]:
        sup, qry = split(samples, args.k, seed, args.rfi_query)
        write_jsonl(f"{OUT}/support_k{args.k}_s{seed}.jsonl", sup)
        write_jsonl(f"{OUT}/query_k{args.k}_s{seed}.jsonl", qry)
        print(f"  seed {seed}: support={len(sup)} query={len(qry)} (pul {sum(1 for s in qry if s['label']=='pulsar')}/rfi {sum(1 for s in qry if s['label']=='rfi')})")


if __name__ == "__main__":
    main()
