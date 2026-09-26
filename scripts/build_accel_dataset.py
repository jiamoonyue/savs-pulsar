"""构建"数据集B（易分类）"：正样本=原GPPS pulsar（复用），负样本=新FAST ACCEL。

输出 data/accel/：
  - pulsar 正样本：复用 region_gpps 的 pulsar，补 text_info（从整图裁）
  - rfi 负样本：裁 ACCEL 新数据集 5 面板 + text_info
  - split 支持集/查询集（pulsar/rfi 各抽 k 支持，剩余做查询）

裁剪坐标：/SAVS_DATA_ROOT/ApJS实验/裁剪坐标.json（5面板 pixel 坐标）
整图源：负=pulsar 从 /CCF-A/datasets/.../PICS（正用现成 crops）；负=新FAST测试/负样本
"""
import argparse, json, os
from PIL import Image

COORD = json.load(open("/SAVS_DATA_ROOT/ApJS实验/裁剪坐标.json"))
PANEL_KEYS = {"profile": "①profile", "time_phase": "②time_phase",
              "freq_phase": "③freq_phase", "dm_curve": "④DM_curve", "text_info": "⑤text_info"}
POS_SRC = "/SAVS_DATA_ROOT/CCF-A/SAVs-main/data/region_gpps/all.jsonl"
POS_CROP = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops"
NEG_SRC = "/SAVS_DATA_ROOT/CCF-A/新FAST测试/负样本"
OUT_PANELS = "/SAVS_DATA_ROOT/ApJS实验/pulsar_clip/artifacts/crops_accel"   # 负样本面板输出


def crop5(img, outdir, base):
    """裁 5 面板存文件，返回 {panel: path}。"""
    paths = {}
    for pk, ckey in PANEL_KEYS.items():
        box = COORD["panels"][ckey]["pixel"]
        crop = img.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
        os.makedirs(outdir, exist_ok=True)
        p = f"{outdir}/{base}_{pk}.png"
        crop.save(p)
        paths[pk] = p
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--rfi_query", type=int, default=2000)
    ap.add_argument("--out", default="data/accel")
    args = ap.parse_args()

    # 1. 正样本：复用 region_gpps pulsar，补 text_info
    pos = [json.loads(l) for l in open(POS_SRC)]
    pos = [x for x in pos if x["label"] == "pulsar"]
    for i, it in enumerate(pos):
        # 从整图源裁 text_info（正样本 crops 有 profile 等，但 text_info 需裁整图——用 pulsar_clip crops 的？没有）
        # 简化：正样本 text_info 从 PICS 整图裁（uid 已在 jsonl? region_gpps 有 uid）
        pass
    print(f"正样本 pulsar: {len(pos)} 张（复用 region_gpps）——text_info 需补裁，见下")


if __name__ == "__main__":
    main()
