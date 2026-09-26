"""SAVs 全局基线（整图输入，无区域感知）——用于 B/Parkes 数据集。

对照实验：region-aware（区域感知）vs 全局 SAVs（整图全局评分+top-k+多数投票）。
对数据集 B（ACCEL）和 Parkes 用整图 jsonl 跑 SAVs 全局。

输入：各数据集的 query jsonl（含 uid/label，整图路径由 dataset 决定）。
- B: /SAVS_DATA_ROOT/CCF-A/新FAST测试/{正,负}样本/*.png（uid 需映射）
- Parkes: data/parkes 的 query jsonl（uid=文件名）→ 整图路径

用法: python scripts/savs_global_baseline.py --dataset B|parkes --k 25
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tqdm import tqdm
import torch
torch.set_grad_enabled(False)
from src.utils import load_model, mllm_encode, mllm_classify

B_POS = "/SAVS_DATA_ROOT/CCF-A/新FAST测试/正样本"
B_NEG = "/SAVS_DATA_ROOT/CCF-A/新FAST测试/负样本"
PARKES_PUL = "/SAVS_DATA_ROOT/CCF-A/datasets/Parkes/sgan_lowlat_dataset_png/pulsars"
PARKES_RFI = "/SAVS_DATA_ROOT/CCF-A/datasets/Parkes/sgan_lowlat_dataset_png/nonpulsars"


def whole_path_B(uid, label):
    # B 的 uid: bpos{idx} / bneg{idx} -> 目录里索引（0-based 序号）
    if uid.startswith("bpos"):
        idx = int(uid[4:])
        fs = sorted(os.listdir(B_POS))
        return f"{B_POS}/{fs[idx]}"
    else:
        idx = int(uid[4:])
        fs = sorted(os.listdir(B_NEG))
        return f"{B_NEG}/{fs[idx]}"


def whole_path_parkes(uid, label):
    dirp = PARKES_PUL if label == "pulsar" else PARKES_RFI
    return f"{dirp}/{uid}"


def build_jsonl_items(prefix, k, dataset):
    """读 dataset 的 support/query jsonl，转成 {image:整图, question:"", label}。"""
    if dataset == "B":
        su = f"/SAVS_DATA_ROOT/CCF-A/新FAST测试/support_k{k}_s0.jsonl"
        qu = f"/SAVS_DATA_ROOT/CCF-A/新FAST测试/query_k{k}_s0.jsonl"
        wp = whole_path_B
    else:
        su = f"data/parkes/support_k{k}_s0.jsonl"
        qu = f"data/parkes/query_k{k}_s0.jsonl"
        wp = whole_path_parkes
    def load(p):
        return [{"image": wp(json.loads(l)["uid"], json.loads(l)["label"]),
                 "question": "", "label": json.loads(l)["label"]} for l in open(p)]
    return load(su), load(qu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["B", "parkes"])
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--num_heads", type=int, default=20)
    args = ap.parse_args()

    train, test = build_jsonl_items("", args.k, args.dataset)
    model = load_model("qwen2.5_vl", "pulsar")
    emb = mllm_encode(model, train, num_head=args.num_heads)
    tp = fp = fn = 0
    for it in tqdm(test, desc="  分类"):
        pred = mllm_classify(it, model, emb)
        if it["label"] == "pulsar":
            if pred == "pulsar": tp += 1
            else: fn += 1
        else:
            if pred == "pulsar": fp += 1
    P = tp/(tp+fp) if tp+fp > 0 else 0; R = tp/(tp+fn) if tp+fn > 0 else 0
    F = 2*P*R/(P+R) if P+R > 0 else 0
    print(f"== SAVs全局 [{args.dataset} k={args.k} heads={args.num_heads}] ==")
    print(f"F1={F:.4f} P={P:.4f} R={R:.4f} (tp={tp} fp={fp} fn={fn})")
    with open(f"results/savs_global_{args.dataset}_k{args.k}.json", "w") as f:
        json.dump({"f1": F, "precision": P, "recall": R, "tp": tp, "fp": fp, "fn": fn}, f, indent=2)


if __name__ == "__main__":
    main()
