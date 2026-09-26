"""区域感知 SAVs：子图分解 → 区域感知头评分 → 跨子图覆盖约束选头 → 分组一致性投票。

对应创新点：
  [1] 区域感知评分   —— 每个头在每个面板上独立算判别力（vs 基线全局单一评分）
  [2] 覆盖约束选头   —— 头必须至少在 Q 个面板上判别力 > 0.5 才入选（防偏科）
  [3] 分组一致性投票 —— 头按主导面板分组，组内多数 + 跨组一致性 conf_cross

用法:
  python scripts/run_regionaware.py --k 25 --num_heads 20 --q 2 --seeds 0 --limit 0
  --limit N>0 只评估前 N 个查询样本（快速验证用）
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from tqdm import tqdm

torch.set_grad_enabled(False)

from metrics_util import confusion, pr_auc, recall_at_fpr, pr_curve  # noqa: E402
from src.utils import load_model, get_last_mean_head_activations

PANELS = ["freq_phase", "time_phase", "dm_curve", "profile"]  # 与 build_region_gpps_data.py 一致
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "region_gpps")

# Prompt 变体（论文做法：任务定义文本，引导 LMM 把图像+文本融合到末尾 token）
PROMPT_VARIANTS = {
    "empty": "",   # 免文本基线（当前默认）
    "task": "Is this a pulsar candidate? Answer Yes or No.",          # 论文风格二分类疑问句
    "neutral": "Classify this pulsar diagnostic plot.",               # 中性指令对照
    "panel": "PANEL_SPECIFIC",  # 特殊值：每面板用自己的物理语义 prompt（见 PROMPT_PER_PANEL）
}
# 面板感知 prompt：输入形态与任务描述对齐（单面板前向 → 该面板的物理语义问题）
# 物理依据（Pulsar astronomy 标准判据）:
#   profile:   平均轮廓可为单峰/双峰/多峰（Rankin 分类）→ "one or more"
#   dm_curve:  真信号在非零 DM 出现对称单峰；RFI 峰在 DM=0 → "non-zero"
#   freq_phase:真信号宽带+色散斜扫；RFI 窄带 → "broadband"
#   time_phase:真信号固定相位、贯穿观测时间的一致性脉冲结构 → "consistent phase"
# 刻意不含任何类名（无 "pulsar"/"candidate"/"RFI" 字样），避免文本类名泄漏到特征
PROMPT_PER_PANEL = {
    "freq_phase": "Does this frequency-phase panel show a broadband, dispersed signal swept diagonally across frequencies? Answer Yes or No.",
    "time_phase": "Does this time-phase panel show a pulse-like structure at a consistent phase across the observation time? Answer Yes or No.",
    "dm_curve": "Does this dispersion measure curve show a single prominent peak at a non-zero DM value? Answer Yes or No.",
    "profile": "Does this folded pulse profile show one or more distinct peaks at stable pulse phases? Answer Yes or No.",
}


def extract_panel_vector(model, panel_path, prompt=""):
    """单面板前向，返回该面板 784 头在最后 token 的向量 [784, 128]。"""
    item = {"image": panel_path, "question": prompt, "label": "x"}
    act = get_last_mean_head_activations([item], model, N_TRIALS=1, shot=0)  # [28, 28, 1, 128]
    return act[:, :, -1, :].reshape(-1, 128)  # [784, 128]


def extract_sample(model, sample, prompt="", per_panel_prompt=None):
    """样本的 4 面板特征 [4, 784, 128]。per_panel_prompt 提供时每面板用各自 prompt。"""
    vecs = []
    for p in PANELS:
        p_prompt = prompt
        if per_panel_prompt is not None:
            p_prompt = per_panel_prompt[p]
        vecs.append(extract_panel_vector(model, sample["panels"][p], p_prompt))
    return torch.stack(vecs)


def region_head_scoring(feats, labels, mode="insample"):
    """feats: [N, 4, 784, 128]; labels: [N]。
    返回 score [784, 4]：每头每面板的类判别力。

    mode="insample": 质心用全部支持集算，评估也在同一集合上（原始实现，系统性高估）。
    mode="loo"     : 留一（leave-one-out）——给第 i 个样本打分时质心只用其余 N-1 个算。
    """
    n, n_panels, n_heads, _ = feats.shape
    lab = torch.tensor(labels, dtype=torch.long, device=feats.device)
    scores = torch.zeros(n_heads, n_panels)
    for r in range(n_panels):
        fr = feats[:, r]  # [N, 784, 128]
        if mode == "loo":
            n0 = int((lab == 0).sum())
            n1 = int((lab == 1).sum())
            sum0 = fr[lab == 0].sum(0)
            sum1 = fr[lab == 1].sum(0)
            correct = torch.zeros(n_heads, device=feats.device)
            for i in range(n):
                if int(lab[i]) == 0:
                    c0 = (sum0 - fr[i]) / max(n0 - 1, 1)
                    c1 = sum1 / max(n1, 1)
                else:
                    c0 = sum0 / max(n0, 1)
                    c1 = (sum1 - fr[i]) / max(n1 - 1, 1)
                v = fr[i]
                sim0 = torch.nn.functional.cosine_similarity(v, c0, dim=-1)
                sim1 = torch.nn.functional.cosine_similarity(v, c1, dim=-1)
                correct += ((sim1 > sim0).long() == int(lab[i])).float()
            scores[:, r] = correct / n
        else:
            c0 = fr[lab == 0].mean(0)
            c1 = fr[lab == 1].mean(0)
            sim0 = torch.nn.functional.cosine_similarity(fr, c0.unsqueeze(0), dim=-1)
            sim1 = torch.nn.functional.cosine_similarity(fr, c1.unsqueeze(0), dim=-1)
            pred = (sim1 > sim0).long()
            scores[:, r] = (pred == lab.unsqueeze(1)).float().mean(0)
    return scores


def select_heads(scores, q=2, k=20):
    """覆盖约束选头：valid_regions >= Q，按有效区域平均分取 top-k。
    返回 (head_indices [k], dominant_panels [k])。候选不足时自动放宽 Q。
    主导面板只在有效(>0.5)区域内取 argmax，避免选中 <0.5 的无效面板当分类面板。"""
    n_heads, n_panels = scores.shape
    valid = scores > 0.5
    n_valid = valid.sum(1)
    best_q = q
    while (n_valid >= best_q).sum() < k and best_q > 1:
        best_q -= 1
        print(f"  候选头不足 top-{k}，覆盖约束放宽到 Q={best_q}")
    mask = (n_valid >= best_q).nonzero().flatten()
    avg = (scores * valid).sum(1) / n_valid.clamp(min=1)
    ranked = avg[mask].argsort(descending=True)
    selected = mask[ranked[:k]]
    # 主导面板只在 valid(>0.5) 面板内取 argmax；valid 里对每个入选头取分最高者
    dominant = torch.zeros(len(selected), dtype=torch.long)
    for i, h in enumerate(selected.tolist()):
        valid_panels = valid[h].nonzero().flatten()   # 有效面板索引
        if len(valid_panels) > 0:
            dominant[i] = valid_panels[scores[h][valid_panels].argmax()]
        else:
            dominant[i] = scores[h].argmax()          # 理论不达，兜底全局 argmax
    return selected, dominant


def build_centroids(support_feats, support_labels, heads, dominant_panels):
    """入选头在其主导面板上的类质心 {head_idx: {label: [128]}}。"""
    lab = torch.tensor(support_labels, dtype=torch.long, device=support_feats.device)
    centroids = {}
    for j, h in enumerate(heads.tolist()):
        r = int(dominant_panels[j])
        fr = support_feats[:, r, h]  # [N, 128]
        c0 = fr[lab == 0].mean(0)
        c1 = fr[lab == 1].mean(0)
        centroids[h] = (c0, c1)
    return centroids


def classify_sample_v2(feat, heads, dominant_panels, centroids, weights=None, method="a1"):
    """feat: [4, k, 128]（缓存布局，面板 s×入选头位置 j）。同 classify_sample 的投票逻辑。"""
    heads_list = heads.tolist()
    domin = dominant_panels.tolist()
    votes = []
    for j, h in enumerate(heads_list):
        v = feat[j]                                # 缓存布局 [k, 128]：每头只存主导面板向量
        c0, c1 = centroids[h]
        s0 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c0.unsqueeze(0))
        s1 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c1.unsqueeze(0))
        votes.append(1 if s1 > s0 else 0)

    if method == "a2" and weights is not None:
        group_weights = {}
        for r in range(len(PANELS)):
            idx = [i for i, dr in enumerate(domin) if dr == r]
            if idx:
                w0 = sum(weights[i] for i in idx if votes[i] == 0)
                w1 = sum(weights[i] for i in idx if votes[i] == 1)
                group_weights[r] = 1 if w1 >= w0 else 0
        w0 = sum(weights[i] for i in range(len(votes)) if votes[i] == 0)
        w1 = sum(weights[i] for i in range(len(votes)) if votes[i] == 1)
        final = 1 if w1 >= w0 else 0
    else:
        group_weights = {}
        for r in range(len(PANELS)):
            idx = [i for i, dr in enumerate(domin) if dr == r]
            if idx:
                group_weights[r] = Counter(votes[i] for i in idx).most_common(1)[0][0]
        final = Counter(votes).most_common(1)[0][0]

    active = list(group_weights.values())
    conf_cross = sum(1 for gv in active if gv == final) / len(active) if active else 1.0
    return votes, final, conf_cross


def classify_sample(feat, heads, dominant_panels, centroids, weights=None, method="a1"):
    """feat: [4, 784, 128]。返回 (投票列表, 最终标签, conf_cross)。

    method="a1" 平权多数投票；method="a2" 评分加权投票（免训练 MoE 路由）：
    weights[j] = 支持集上 head_j 在主导面板的判别力评分，高判别力头权重更大。
    """
    heads_list = heads.tolist()
    domin = dominant_panels.tolist()
    votes = []
    for j, h in enumerate(heads_list):
        r = int(domin[j])
        v = feat[r, h]
        c0, c1 = centroids[h]
        s0 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c0.unsqueeze(0))
        s1 = torch.nn.functional.cosine_similarity(v.unsqueeze(0), c1.unsqueeze(0))
        votes.append(1 if s1 > s0 else 0)

    if method == "a2" and weights is not None:
        # 组内加权多数
        group_weights = {}
        for r in range(len(PANELS)):
            idx = [i for i, dr in enumerate(domin) if dr == r]
            if idx:
                w0 = sum(weights[i] for i in idx if votes[i] == 0)
                w1 = sum(weights[i] for i in idx if votes[i] == 1)
                group_weights[r] = 1 if w1 >= w0 else 0
        # 全体加权多数
        w0 = sum(weights[i] for i in range(len(votes)) if votes[i] == 0)
        w1 = sum(weights[i] for i in range(len(votes)) if votes[i] == 1)
        final = 1 if w1 >= w0 else 0
    else:
        group_weights = {}
        for r in range(len(PANELS)):
            idx = [i for i, dr in enumerate(domin) if dr == r]
            if idx:
                group_weights[r] = Counter(votes[i] for i in idx).most_common(1)[0][0]
        final = Counter(votes).most_common(1)[0][0]

    active = list(group_weights.values())
    conf_cross = sum(1 for gv in active if gv == final) / len(active) if active else 1.0
    return votes, final, conf_cross


def extract_support_features(model, support, cache_path=None, prompt="", per_panel_prompt=None):
    """支持集特征 [N, 4, 784, 128]；有缓存则加载，无则提取并保存。"""
    if cache_path and os.path.exists(cache_path):
        print(f"  加载支持集特征缓存: {cache_path}")
        return torch.load(cache_path)
    print("  提取支持集面板特征...")
    feats = torch.stack([extract_sample(model, s, prompt, per_panel_prompt) for s in tqdm(support, desc="  支持集")])
    if cache_path:
        torch.save(feats, cache_path)
        print(f"  支持集特征已缓存: {cache_path}")
    return feats


def extract_query_features(model, query, heads, dominant, num_heads, cache_path=None,
                           limit=0, prompt="", per_panel_prompt=None):
    """查询集 top-头特征 [M, k, 128]（每头只存主导面板向量）；有缓存则加载。

    单样本串行前向（正式路径）：全部历史数字与此一致。
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  加载查询集特征缓存: {cache_path}")
        return torch.load(cache_path)
    heads_list = heads.tolist()
    domin = dominant.tolist()
    print("  提取查询集特征（只取入选头）...")
    feats = []
    for s in tqdm(query, desc="  查询集"):
        full = extract_sample(model, s, prompt, per_panel_prompt)  # [4, 784, 128]
        sub = torch.stack([full[r, h] for j, (h, r) in enumerate(zip(heads_list, domin))])
        feats.append(sub)                          # [k, 128]
    feats = torch.stack(feats)                     # [M, k, 128]
    if cache_path:
        torch.save(feats, cache_path)
        print(f"  查询集特征已缓存: {cache_path}")
    return feats


def extract_query_features_full(model, query, cache_path=None, limit=0,
                                prompt="", per_panel_prompt=None):
    """查询集全量头特征 [M, 4, 784, 128]（fp16 存盘 ≈2.5GB/档）。

    头数消融专用：一次提取全量头向量，所有 num_heads 的选择/投票秒级复用。
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  加载查询集全量头缓存: {cache_path}")
        return torch.load(cache_path)
    print("  提取查询集全量头特征...")
    feats = []
    for s in tqdm(query, desc="  查询集"):
        full = extract_sample(model, s, prompt, per_panel_prompt)  # [4, 784, 128]
        feats.append(full.bfloat16())                # bf16 与模型原始输出同 dtype（逐位一致）
    feats = torch.stack(feats)                     # [M, P, 784, 128]
    if cache_path:
        # CHUNKED_SAVE: 大文件在部分网络文件系统上会被静默截断
        # （实测 ACCEL 的 2.42GB 缓存两次都截断在同一字节）。故超阈值时分块保存。
        _nbytes = feats.numel() * feats.element_size()
        _LIMIT = 1_800_000_000
        if _nbytes <= _LIMIT:
            torch.save(feats, cache_path)
            print(f"  查询集全量头特征已缓存: {cache_path} ({_nbytes/1e9:.2f} GB)")
        else:
            _nchunk = int(_nbytes // _LIMIT) + 1
            _per = (feats.shape[0] + _nchunk - 1) // _nchunk
            for _ci in range(_nchunk):
                _part = feats[_ci * _per:(_ci + 1) * _per]
                if _part.shape[0] == 0:
                    continue
                _cp = f"{cache_path}.chunk{_ci}.pt"
                # ⚠ 必须 .clone()：_part 是切片【视图】，torch.save 对视图会存下
                # 整个底层 storage（实测 1.209GB 的切片被存成 2.418GB，两块尺寸相同）。
                # .contiguous() 无效（已连续，返回自身）。
                torch.save(_part.clone(), _cp)
                print(f"  分块缓存 [{_ci}] {_cp}  {tuple(_part.shape)}")
            print(f"  查询集全量头特征已分块缓存: {cache_path}.chunk*({_nchunk})")
    return feats


def evaluate(model, support_path, query_path, num_heads, q, limit=0, method="a1",
             cache_dir=None, prompt="", prompt_tag="empty", per_panel_prompt=None,
             store_full=False, score_mode="insample", support_only=False):  # SUPPORT_ONLY_PATCH
    with open(support_path) as fh:
        support = [json.loads(line) for line in fh]
    with open(query_path) as fh:
        query = [json.loads(line) for line in fh]
    if limit > 0:
        query = query[:limit]

    seg = os.path.basename(os.path.dirname(support_path))  # region
    cache_prefix = os.path.basename(support_path).replace(".jsonl", "")  # support_k25_s0
    # prompt 改变输入序列 → 特征全变，缓存必须按 prompt 变体隔离（empty/task/neutral/panel）
    sup_cache = os.path.join(cache_dir, f"{seg}_{cache_prefix}_{prompt_tag}_supfeat.pt") if cache_dir else None
    if store_full:
        # 全量头缓存与 num_heads/Q 无关（存全部 784 头）→ 不含 h/q，run_head_ablation.py 直接读
        qry_cache = os.path.join(cache_dir, f"{seg}_{cache_prefix}_{prompt_tag}_fullqryfeat.pt") if cache_dir else None
    else:
        # ⚠️ top-k 缓存：选头/主导面板依赖 num_heads/Q → 不含它们会错位加载
        qry_cache = (os.path.join(cache_dir, f"{seg}_{cache_prefix}_{prompt_tag}_h{num_heads}_q{q}_qryfeat.pt")
                     if cache_dir else None)
    if limit > 0:
        # --limit 只测部分样本：绝不允许读写缓存（残缺缓存会污染后续全量）
        qry_cache = None

    support_feats = extract_support_features(model, support, sup_cache, prompt, per_panel_prompt)
    support_labels = [0 if s["label"] == "rfi" else 1 for s in support]

    print("  区域感知头评分 + 覆盖约束选头...")
    scores = region_head_scoring(support_feats, torch.tensor(support_labels), mode=score_mode)
    heads, dominant = select_heads(scores, q=q, k=num_heads)
    print("  入选头 (head_idx, 主导面板, 面板分数):")
    for j, h in enumerate(heads.tolist()):
        r = int(dominant[j])
        print(f"    head {h:4d} -> {PANELS[r]:11s} score={scores[h].tolist()}")
    centroids = build_centroids(support_feats, torch.tensor(support_labels), heads, dominant)

    # SUPPORT_ONLY_PATCH: 只提取支持集全量头并早退（查询特征是 k/seed 无关的，可复用）
    if support_only:
        return None

    # 查询集特征（含缓存；store_full 时存全量头供头数消融复用）
    if store_full:
        extract_query_features_full(model, query, qry_cache, limit, prompt, per_panel_prompt)
        return None     # 仅提取+缓存，不做评估（由 run_head_ablation.py 复用）
    query_feats = extract_query_features(model, query, heads, dominant, num_heads,
                                         qry_cache, limit, prompt, per_panel_prompt)
    weights = [scores[h, r].item() for h, r in zip(heads.tolist(), dominant.tolist())]

    print(f"  投票分类 method={method} ...")
    tp = fp = fn = 0
    cross_low = 0
    _margins = []   # DUMP_SCORES_V1
    _finals = []
    for idx, s in enumerate(tqdm(query, desc="  分类")):
        # query_feats[idx]: [k, 128]（每头在主导面板上的向量）
        feat = query_feats[idx]
        votes, final, conf_cross = classify_sample_v2(feat, heads, dominant, centroids,
                                                      weights, method)
        _n1 = int(sum(votes))
        _margins.append(_n1 - (len(votes) - _n1))
        _finals.append(int(final))
        y = 1 if s["label"] == "pulsar" else 0
        if conf_cross < 0.6:
            cross_low += 1
        if y == 1:
            if final == 1:
                tp += 1
            else:
                fn += 1
        else:
            if final == 1:
                fp += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    acc = (tp + (len(query) - tp - fp - fn)) / len(query)
    print(f"  F1={f1:.4f} P={p:.4f} R={r:.4f} Acc={acc:.4f} | 低一致性样本: {cross_low}/{len(query)}")
    # 附带入选头信息（可视化用）：layer/head/主导面板/面板分数
    heads_info = [{"layer": int(hl), "head": int(hh),
                   "dominant_panel": PANELS[int(dom)],
                   "scores": scores[int(hh)].tolist(),
                   "dominant_score": scores[int(hh), int(dom)].item()}
                  for hl, hh, dom in zip(heads // 28, heads % 28, dominant)]
    # DUMP_SCORES_V1 自检：用落盘的 finals 反算 tp/fp/fn，必须等于计数器
    _tp = sum(1 for pr, sm in zip(_finals, query) if pr == 1 and sm["label"] == "pulsar")
    _fp = sum(1 for pr, sm in zip(_finals, query) if pr == 1 and sm["label"] != "pulsar")
    _fn = sum(1 for pr, sm in zip(_finals, query) if pr == 0 and sm["label"] == "pulsar")
    assert (_tp, _fp, _fn) == (tp, fp, fn), (
        "DUMP_SCORES_V1 自检失败: finals 反算 (%d,%d,%d) != 计数器 (%d,%d,%d)" % (_tp, _fp, _fn, tp, fp, fn))
    _labs = [1 if s["label"] == "pulsar" else 0 for s in query]
    _extra = {
        "confusion": confusion(_finals, _labs),
        "pr_auc": pr_auc(_margins, _labs),
        "recall_at_fpr": recall_at_fpr(_margins, _labs),
        "pr_curve": pr_curve(_margins, _labs),
        "n_pos": int(sum(_labs)),
        "n_neg": int(len(_labs) - sum(_labs)),
    }
    return {"f1": f1, "precision": p, "recall": r, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "n_query": len(query), "n_support": len(support),
            "heads": heads_info, "margins": _margins, "finals": _finals, **_extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_heads", type=int, default=20)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--q", type=int, default=2, help="覆盖约束：至少 Q 个面板有效")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--method", type=str, default="a1", choices=["a1", "a2"],
                    help="a1=平权多数投票; a2=评分加权投票(免训练MoE路由)")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评估前 N 个查询样本")
    ap.add_argument("--cache_dir", type=str, default=None, help="特征缓存目录（复用免重跑前向）")
    ap.add_argument("--prompt_type", type=str, default="empty", choices=list(PROMPT_VARIANTS.keys()),
                    help="prompt 变体: empty/task/neutral/panel")
    ap.add_argument("--store_full", action="store_true",
                    help="只提取查询集全量头特征并缓存（供头数消融），不做评估")
    ap.add_argument("--score_mode", type=str, default="insample", choices=["insample", "loo"],
                    help="选头打分: insample=原始(高估); loo=留一(无偏)")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="数据目录(region_gpps 纯GPPS/region 混数据)，默认用 DATA_DIR")
    ap.add_argument("--model_name", type=str, default="qwen2.5_vl",
                    help="模型名: qwen2.5_vl(7B) / qwen2.5_vl_3b(3B)，骨架消融用")
    ap.add_argument("--model_tag", type=str, default="qwen25_7b",
                    help="模型标记，区分骨架（如 qwen25_3b），防结果文件互相覆盖")
    ap.add_argument("--panels", type=str, default=None,
                    help="逗号分隔面板名，覆盖模块级 PANELS（HTRU1 为 3 面板）")
    # MULTIDATA_PATCH
    args = ap.parse_args()
    global PANELS
    if args.panels:
        PANELS = [p.strip() for p in args.panels.split(",") if p.strip()]
        print("  面板覆盖为: %s" % PANELS)
    seeds = [int(x) for x in args.seeds.split(",")]
    prompt = PROMPT_VARIANTS[args.prompt_type]
    per_panel_prompt = PROMPT_PER_PANEL if args.prompt_type == "panel" else None
    global DATA_DIR
    if args.data_dir:
        # 支持绝对路径（数据集B在新FAST测试）或相对 data/ 子目录
        DATA_DIR = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(
            os.path.dirname(__file__), "..", "data", args.data_dir)

    model = load_model(args.model_name, "pulsar")
    os.makedirs(args.cache_dir, exist_ok=True) if args.cache_dir else None
    all_results = {}
    for seed in seeds:
        sup = os.path.join(DATA_DIR, f"support_k{args.k}_s{seed}.jsonl")
        qry = os.path.join(DATA_DIR, f"query_k{args.k}_s{seed}.jsonl")
        if not os.path.exists(sup):
            print(f"缺少 {sup}，先运行 build_region_gpps_data.py --k {args.k} --seed {seed}")
            continue
        print(f"\n=== seed {seed} ===")
        r = evaluate(model, sup, qry, args.num_heads, args.q, args.limit,
                     args.method, args.cache_dir, prompt, args.prompt_type,
                     per_panel_prompt, args.store_full, args.score_mode)
        if r is None:
            continue     # store_full 模式只提取特征缓存
        all_results[f"s{seed}"] = r

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    data_tag = os.path.basename(DATA_DIR)  # region_gpps / region / 新FAST测试
    model_tag = getattr(args, "model_tag", "qwen25_7b")  # 区分骨架（防覆盖）
    out = os.path.join(out_dir,
                       f"{data_tag}_k{args.k}_h{args.num_heads}_q{args.q}_{args.method}_p{args.prompt_type}_{model_tag}.json")
    # GUARD_NOWRITE: 绝不用空结果覆盖既有文件（曾把 FAST merged 结果写成 {}）
    if getattr(args, "store_full", False):
        print("[跳过写结果] store_full 模式只提取特征缓存")
    elif not all_results:
        print("[跳过写结果] all_results 为空，拒绝覆盖 %s" % out)
    else:
        with open(out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print("结果已保存: %s" % out)
    if len(all_results) > 1:
        f1s = np.array([r["f1"] for r in all_results.values()])
        ps = np.array([r["precision"] for r in all_results.values()])
        rs = np.array([r["recall"] for r in all_results.values()])
        print(f"== 汇总 (k={args.k}, heads={args.num_heads}, Q={args.q}, {len(all_results)} seeds) ==")
        print(f"F1 mean={f1s.mean():.4f} ± {f1s.std():.4f} | P mean={ps.mean():.4f} | R mean={rs.mean():.4f}")


if __name__ == "__main__":
    main()
