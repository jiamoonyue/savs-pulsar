"""从诊断图整图裁出 text_info（Search Information）区块，供 VLM 读数值。

用法: python scripts/extract_textinfo.py --uid u013714,u013715 --out /tmp/textinfo/
需要: 整图源 + 裁剪坐标(text_info)。
"""
import argparse, json, os, re
from PIL import Image

COORD = json.load(open("/SAVS_DATA_ROOT/ApJS实验/裁剪坐标.json"))
# 整图源: PICS 目录（uid -> manifest file -> 路径）
import csv
rows = [r for r in csv.DictReader(open("/SAVS_DATA_ROOT/ApJS实验/FAST_split/manifest_full.csv", encoding="utf-8-sig"))]
rows.sort(key=lambda r: r["file"])
uid2file = {f"u{i:06d}": r for i, r in enumerate(rows)}

PICS = "/SAVS_DATA_ROOT/CCF-A/datasets/FAST/PICS-ResNet_data_png"


def source_image(uid):
    f = uid2file[uid]["file"].replace("\\", "/")  # e.g. test_data/pulsar/xx.png
    # file 形如 test_data\pulsar\FP....png
    rel = f.replace("test_data", "test_data").replace("train_data", "train_data")
    return os.path.join(PICS, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", type=str, required=True, help="逗号分隔的 uid")
    ap.add_argument("--out", default="/tmp/textinfo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    box = COORD["panels"]["⑤text_info"]["pixel"]
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    for uid in args.uid.split(","):
        path = source_image(uid)
        if not os.path.exists(path):
            print(f"[{uid}] 缺整图 {path}")
            continue
        img = Image.open(path).convert("RGB")
        if img.size != (2200, 1700):
            img = img.resize((2200, 1700))
        crop = img.crop((x0, y0, x1, y1))
        crop.save(f"{args.out}/{uid}_textinfo.png")
        print(f"[{uid}] {path} -> {args.out}/{uid}_textinfo.png ({crop.size}) label={uid2file[uid]['label']}")


if __name__ == "__main__":
    main()
