#!/usr/bin/env python3
"""Analyze Stage1 residual/accel/jerk abnormal frame positions and prefix attribution."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_data_processing.loader import (
    episode_parquet_path,
    list_episode_indices,
    read_episode_arrays,
)
from robot_data_processing.mask import (
    build_step_validity_mask,
    compute_per_joint_action_zero_exclude,
    compute_per_joint_stage1_exclude,
)
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml
from robot_data_processing.smoothing import compute_derivatives, hybrid_threshold, smooth_1d
from robot_data_processing.stages.stage1_sudden_change import (
    _cluster_filter,
    _hard_limit_abnormal_frames,
    run_stage1,
)
from robot_data_processing.stages.stage3_extreme_value import run_stage3
from robot_data_processing.types import GlobalStats


def dim_metrics(x: np.ndarray, cfg) -> dict:
    sm = smooth_1d(x, cfg.median_kernel, cfg.savgol_window, cfg.savgol_polyorder)
    res = np.abs(x - sm)
    accel, jerk = compute_derivatives(x)
    a, j = np.abs(accel), np.abs(jerk)
    thr_r = hybrid_threshold(res, cfg.k_residual, cfg.percentile_floor)
    thr_a = hybrid_threshold(a, cfg.k_accel, min(cfg.percentile_floor, 99.95))
    thr_j = hybrid_threshold(j, cfg.k_jerk, min(cfg.percentile_floor, 99.95))
    r_over = res > thr_r
    a_over = a > thr_a
    j_over = j > thr_j
    sudden = r_over & (a_over | j_over)
    return {"r_over": r_over, "a_over": a_over, "j_over": j_over, "sudden": sudden}


def add_position(hist: np.ndarray, T: int, mask: np.ndarray) -> None:
    for t in np.flatnonzero(mask):
        hist[min(int(t / max(T - 1, 1) * 100), 99)] += 1


def episode_frame_flags(sa, aa, se, s1_ex, s1_cfg, xyz):
    T = sa.shape[0]
    r_f = np.zeros(T, dtype=bool)
    a_f = np.zeros(T, dtype=bool)
    j_f = np.zeros(T, dtype=bool)
    s_f = np.zeros(T, dtype=bool)
    sudden_per_dim = []

    for d in range(12):
        m = dim_metrics(sa[:, d], s1_cfg)
        ex = s1_ex[:, d]
        for key, arr in [("r_over", r_f), ("a_over", a_f), ("j_over", j_f), ("sudden", s_f)]:
            flag = m[key].copy()
            flag[ex] = False
            arr |= flag
        sudden_per_dim.append(m["sudden"].copy())
        sudden_per_dim[-1][ex] = False

    for d in range(12):
        m = dim_metrics(aa[:, d], s1_cfg)
        ex = s1_ex[:, d]
        sd = m["sudden"].copy()
        sd[ex] = False
        sudden_per_dim.append(sd)
        for key, arr, ex_apply in [
            ("r_over", r_f, True),
            ("a_over", a_f, True),
            ("j_over", j_f, True),
            ("sudden", s_f, True),
        ]:
            flag = m[key].copy()
            if ex_apply:
                flag[ex] = False
            arr |= flag

    for d in range(6):
        m = dim_metrics(se[:, xyz[d]], s1_cfg)
        sudden_per_dim.append(m["sudden"])
        r_f |= m["r_over"]
        a_f |= m["a_over"]
        j_f |= m["j_over"]
        s_f |= m["sudden"]

    cluster_f = np.zeros(T, dtype=bool)
    for sd in sudden_per_dim:
        cluster_f |= _cluster_filter(sd, s1_cfg.frame_abnormal_min_cluster)

    return r_f, a_f, j_f, s_f, cluster_f


def sudden_component_at_frame(a_f: np.ndarray, j_f: np.ndarray, t: int) -> str:
    ha, hj = a_f[t], j_f[t]
    if ha and hj:
        return "both_accel_jerk"
    if ha:
        return "accel"
    if hj:
        return "jerk"
    return "residual_only"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Stage1 metric abnormal frames")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/exp3000_v8/analysis"),
        help="Directory for plots and JSON report",
    )
    parser.add_argument(
        "--stats-cache",
        type=Path,
        default=Path("output/exp3000_v7/cache/global_stats.npz"),
    )
    parser.add_argument("--episode-limit", type=int, default=3000)
    args = parser.parse_args()

    pipe = pipeline_config_from_yaml(load_config(Path("config/humanoid_merged.yaml")), {})
    s1_cfg = pipe.stage1
    stats = GlobalStats.load(str(args.stats_cache))
    root = pipe.dataset_root
    indices = list_episode_indices(root, args.episode_limit)[: args.episode_limit]
    xyz = [0, 1, 2, 7, 8, 9]
    eps, grace = pipe.action_zero_epsilon, pipe.stage1_post_zero_grace_frames

    pos_hist = {m: np.zeros(100, dtype=np.int64) for m in [
        "residual", "accel", "jerk", "sudden", "sudden_accel", "sudden_jerk", "sudden_both"
    ]}
    frame_counts = Counter()
    prefix_attr = Counter()
    truncated_eps = 0
    sudden_cluster_frames = 0

    for i, ep in enumerate(indices):
        data = read_episode_arrays(episode_parquet_path(root, ep))
        sa = data["observation.state.arm.position"]
        aa = data["action.arm.position"]
        se = data["observation.state.end.position"]
        st, ac = data["observation.state"], data["action"]
        T = sa.shape[0]
        s1_ex = compute_per_joint_stage1_exclude(aa, eps, grace)
        s3_ex = compute_per_joint_action_zero_exclude(aa, eps)

        r_f, a_f, j_f, s_f, cluster_f = episode_frame_flags(sa, aa, se, s1_ex, s1_cfg, xyz)
        frame_counts["residual_exceed"] += int(r_f.sum())
        frame_counts["accel_exceed"] += int(a_f.sum())
        frame_counts["jerk_exceed"] += int(j_f.sum())
        frame_counts["sudden_combined"] += int(s_f.sum())
        sudden_cluster_frames += int(cluster_f.sum())

        for m, arr in [
            ("residual", r_f),
            ("accel", a_f),
            ("jerk", j_f),
            ("sudden", s_f),
        ]:
            add_position(pos_hist[m], T, arr)

        for t in np.flatnonzero(s_f):
            comp = sudden_component_at_frame(a_f, j_f, t)
            if comp == "both_accel_jerk":
                frame_counts["sudden_both"] += 1
                pos_hist["sudden_both"][min(int(t / max(T - 1, 1) * 100), 99)] += 1
            elif comp == "accel":
                frame_counts["sudden_accel"] += 1
                pos_hist["sudden_accel"][min(int(t / max(T - 1, 1) * 100), 99)] += 1
            elif comp == "jerk":
                frame_counts["sudden_jerk"] += 1
                pos_hist["sudden_jerk"][min(int(t / max(T - 1, 1) * 100), 99)] += 1

        s1 = run_stage1(sa, se, aa, s1_cfg, startup_exclude_per_joint=s1_ex)
        s3 = run_stage3(st, ac, se, stats, pipe.stage3, startup_exclude_per_joint=s3_ex)
        abnormal = s1.abnormal_frames | s3.remove_frames
        prefix = int(build_step_validity_mask(T, abnormal).sum())
        if prefix >= T:
            continue

        truncated_eps += 1
        fb = int(np.flatnonzero(abnormal)[0])
        hard, _, _ = _hard_limit_abnormal_frames(sa, se, aa, s1_cfg)

        if hard[fb]:
            prefix_attr["hard_limit"] += 1
        elif s3.remove_frames[fb] and not s1.abnormal_frames[fb]:
            prefix_attr["stage3"] += 1
        elif cluster_f[fb]:
            prefix_attr[sudden_component_at_frame(a_f, j_f, fb)] += 1
        elif s_f[fb]:
            prefix_attr[sudden_component_at_frame(a_f, j_f, fb)] += 1
            prefix_attr["sudden_pre_cluster"] += 1
        elif s3.remove_frames[fb]:
            prefix_attr["stage3"] += 1
        else:
            prefix_attr["unknown"] += 1

        if (i + 1) % 500 == 0:
            print(f"{i + 1}/3000", flush=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    centers = np.arange(0.5, 100, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        ("residual", "Residual > threshold", "#2563eb"),
        ("accel", "Accel > threshold", "#16a34a"),
        ("jerk", "Jerk > threshold", "#dc2626"),
        ("sudden", "Combined: residual & (accel|jerk)", "#7c3aed"),
    ]
    for ax, (m, title, c) in zip(axes.flat, panels):
        h = pos_hist[m].astype(float)
        h = h / h.sum() * 100 if h.sum() else h
        ax.bar(centers, h, width=1.0, color=c, alpha=0.75)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Position in episode (%)")
        ax.set_ylabel("% of flagged frames")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Stage1 metric exceedance · position distribution · 3000 episodes", fontsize=13)
    fig.savefig(out / "stage1_abnormal_position.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for m, label, c in [
        ("sudden_accel", "accel triggers sudden", "#16a34a"),
        ("sudden_jerk", "jerk only triggers sudden", "#dc2626"),
        ("sudden_both", "both accel & jerk", "#ca8a04"),
    ]:
        h = pos_hist[m].astype(float)
        h = h / h.sum() * 100 if h.sum() else h
        ax.plot(centers, h, label=label, color=c, lw=1.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Position in episode (%)")
    ax.set_ylabel("% of sudden frames")
    ax.set_title("Sudden-change frames by triggering component")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "stage1_sudden_position_by_component.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = list(prefix_attr.keys())
    vals = [prefix_attr[k] for k in labels]
    ax.bar(labels, vals, color="#6366f1", edgecolor="none")
    ax.set_ylabel("Episodes")
    ax.set_title(f"First abnormal frame cause · truncated: {truncated_eps}/3000")
    plt.xticks(rotation=25, ha="right")
    for idx, v in enumerate(vals):
        ax.text(idx, v + 5, str(v), ha="center", fontsize=9)
    fig.savefig(out / "stage1_prefix_attribution.png", dpi=160)
    plt.close(fig)

    report = {
        "frame_counts": dict(frame_counts),
        "frame_counts_note": "frame-level OR across dims; arm/action startup exclude applied",
        "sudden_after_cluster_frames": sudden_cluster_frames,
        "truncated_episodes": truncated_eps,
        "prefix_attribution_first_abnormal": dict(prefix_attr),
    }
    with (out / "stage1_abnormal_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
