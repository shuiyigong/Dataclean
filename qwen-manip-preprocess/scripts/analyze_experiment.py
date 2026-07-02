#!/usr/bin/env python3
"""Analyze pipeline experiment results (exclusion log + global Stage1 metrics)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.loader import episode_parquet_path, list_episode_indices, read_episode_arrays
from robot_data_processing.mask import (
    build_step_validity_mask,
    compute_per_joint_action_zero_exclude,
    compute_per_joint_stage1_exclude,
)
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml
from robot_data_processing.smoothing import compute_derivatives, detect_sudden_changes_2d_with_thresholds, smooth_1d
from robot_data_processing.stage1_stats import Stage1GlobalStats
from robot_data_processing.stages.stage1_sudden_change import (
    _cluster_filter,
    _hard_limit_abnormal_frames,
    run_stage1,
)
from robot_data_processing.stages.stage3_extreme_value import run_stage3
from robot_data_processing.types import GlobalStats

EE_XYZ = [0, 1, 2, 7, 8, 9]


def dim_metrics_global(x: np.ndarray, thr_r: float, thr_a: float, thr_j: float, cfg) -> dict:
    sm = smooth_1d(x, cfg.median_kernel, cfg.savgol_window, cfg.savgol_polyorder)
    res = np.abs(x - sm)
    accel, jerk = compute_derivatives(x)
    a, j = np.abs(accel), np.abs(jerk)
    r_over = res > thr_r
    a_over = a > thr_a
    j_over = j > thr_j
    return {"r_over": r_over, "a_over": a_over, "j_over": j_over, "sudden": r_over & (a_over | j_over)}


def episode_frame_flags_global(sa, aa, se, s1_ex, s1_cfg, s1_stats: Stage1GlobalStats):
    T = sa.shape[0]
    r_f = np.zeros(T, dtype=bool)
    a_f = np.zeros(T, dtype=bool)
    j_f = np.zeros(T, dtype=bool)
    s_f = np.zeros(T, dtype=bool)
    sudden_per_dim = []

    sk = dict(
        median_kernel=s1_cfg.median_kernel,
        savgol_window=s1_cfg.savgol_window,
        savgol_polyorder=s1_cfg.savgol_polyorder,
    )

    arm_sudden = detect_sudden_changes_2d_with_thresholds(
        sa, s1_stats.thr_residual, s1_stats.thr_accel, s1_stats.thr_jerk,
        channel_offset=0, **sk,
    )
    for d in range(12):
        ch = d
        m = dim_metrics_global(
            sa[:, d], float(s1_stats.thr_residual[ch]), float(s1_stats.thr_accel[ch]),
            float(s1_stats.thr_jerk[ch]), s1_cfg,
        )
        ex = s1_ex[:, d]
        for key, arr in [("r_over", r_f), ("a_over", a_f), ("j_over", j_f), ("sudden", s_f)]:
            flag = m[key].copy()
            flag[ex] = False
            arr |= flag
        sd = arm_sudden[:, d].copy()
        sd[ex] = False
        sudden_per_dim.append(sd)

    action_sudden = detect_sudden_changes_2d_with_thresholds(
        aa, s1_stats.thr_residual, s1_stats.thr_accel, s1_stats.thr_jerk,
        channel_offset=18, **sk,
    )
    for d in range(12):
        ch = 18 + d
        m = dim_metrics_global(
            aa[:, d], float(s1_stats.thr_residual[ch]), float(s1_stats.thr_accel[ch]),
            float(s1_stats.thr_jerk[ch]), s1_cfg,
        )
        ex = s1_ex[:, d]
        for key, arr in [("r_over", r_f), ("a_over", a_f), ("j_over", j_f), ("sudden", s_f)]:
            flag = m[key].copy()
            flag[ex] = False
            arr |= flag
        sd = action_sudden[:, d].copy()
        sd[ex] = False
        sudden_per_dim.append(sd)

    ee_sudden = detect_sudden_changes_2d_with_thresholds(
        se[:, EE_XYZ], s1_stats.thr_residual, s1_stats.thr_accel, s1_stats.thr_jerk,
        channel_offset=12, **sk,
    )
    for d in range(6):
        ch = 12 + d
        m = dim_metrics_global(
            se[:, EE_XYZ[d]], float(s1_stats.thr_residual[ch]), float(s1_stats.thr_accel[ch]),
            float(s1_stats.thr_jerk[ch]), s1_cfg,
        )
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


def add_position(hist: np.ndarray, T: int, mask: np.ndarray) -> None:
    for t in np.flatnonzero(mask):
        hist[min(int(t / max(T - 1, 1) * 100), 99)] += 1


def analyze_exclusion_log(log_path: Path) -> dict:
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    truncated = [r for r in rows if r["first_abnormal_frame"] is not None]
    kept_ratios = [r["kept_frames"] / r["num_frames"] for r in rows]
    prefix_pcts = [
        100 * r["first_abnormal_frame"] / max(r["num_frames"] - 1, 1) for r in truncated
    ]
    da = [r["stage2_da_mean"] for r in rows if r["stage2_da_mean"] is not None]

    return {
        "total_episodes": len(rows),
        "truncated_episodes": len(truncated),
        "fully_kept_episodes": len(rows) - len(truncated),
        "truncation_rate": len(truncated) / len(rows),
        "kept_frame_ratio": {
            "mean": float(np.mean(kept_ratios)),
            "median": float(np.median(kept_ratios)),
            "min": float(np.min(kept_ratios)),
            "p10": float(np.percentile(kept_ratios, 10)),
            "p25": float(np.percentile(kept_ratios, 25)),
        },
        "first_abnormal_position_pct": {
            "mean": float(np.mean(prefix_pcts)) if prefix_pcts else None,
            "median": float(np.median(prefix_pcts)) if prefix_pcts else None,
        },
        "stage1_flagged_episodes": sum(1 for r in rows if r["stage1_flagged_frames"] > 0),
        "stage1_flagged_frames_total": sum(r["stage1_flagged_frames"] for r in rows),
        "stage3_excluded_frames_total": sum(r["stage3_excluded_frames"] for r in rows),
        "stage2_da": {
            "mean": float(np.mean(da)),
            "min": float(np.min(da)),
            "max": float(np.max(da)),
            "below_0.65": sum(1 for x in da if x < 0.65),
            "below_0.7": sum(1 for x in da if x < 0.7),
        },
        "frames_lost_by_truncation": sum(r["num_frames"] - r["kept_frames"] for r in rows),
    }


def analyze_stage1_global(
    pipe,
    indices: list[int],
    s3_stats: GlobalStats,
    s1_stats: Stage1GlobalStats,
) -> dict:
    s1_cfg = pipe.stage1
    root = pipe.dataset_root
    eps, grace = pipe.action_zero_epsilon, pipe.stage1_post_zero_grace_frames

    pos_hist = {m: np.zeros(100, dtype=np.int64) for m in [
        "residual", "accel", "jerk", "sudden", "sudden_accel", "sudden_jerk", "sudden_both"
    ]}
    frame_counts: Counter = Counter()
    prefix_attr: Counter = Counter()
    truncated_eps = 0

    for i, ep in enumerate(indices):
        data = read_episode_arrays(episode_parquet_path(root, ep))
        sa = data["observation.state.arm.position"]
        aa = data["action.arm.position"]
        se = data["observation.state.end.position"]
        st, ac = data["observation.state"], data["action"]
        T = sa.shape[0]
        s1_ex = compute_per_joint_stage1_exclude(aa, eps, grace)
        s3_ex = compute_per_joint_action_zero_exclude(aa, eps)

        r_f, a_f, j_f, s_f, cluster_f = episode_frame_flags_global(sa, aa, se, s1_ex, s1_cfg, s1_stats)
        frame_counts["residual_exceed"] += int(r_f.sum())
        frame_counts["accel_exceed"] += int(a_f.sum())
        frame_counts["jerk_exceed"] += int(j_f.sum())
        frame_counts["sudden_combined"] += int(s_f.sum())

        for m, arr in [("residual", r_f), ("accel", a_f), ("jerk", j_f), ("sudden", s_f)]:
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

        s1 = run_stage1(sa, se, aa, s1_cfg, startup_exclude_per_joint=s1_ex, global_stats=s1_stats)
        s3 = run_stage3(st, ac, se, s3_stats, pipe.stage3, startup_exclude_per_joint=s3_ex)
        abnormal = s1.abnormal_frames | s3.remove_frames
        prefix = int(build_step_validity_mask(T, abnormal).sum())
        if prefix >= T:
            continue

        truncated_eps += 1
        fb = int(np.flatnonzero(abnormal)[0])
        hard, _, _ = _hard_limit_abnormal_frames(sa, se, aa, s1_cfg)

        if hard[fb]:
            prefix_attr["hard_limit"] += 1
        elif s1.abnormal_frames[fb] and not s3.remove_frames[fb]:
            prefix_attr[sudden_component_at_frame(a_f, j_f, fb)] += 1
        elif s3.remove_frames[fb] and not s1.abnormal_frames[fb]:
            prefix_attr["stage3"] += 1
        elif s1.abnormal_frames[fb] and s3.remove_frames[fb]:
            prefix_attr["stage1_and_stage3"] += 1
        else:
            prefix_attr["unknown"] += 1

        if (i + 1) % 500 == 0:
            print(f"  stage1 scan {i + 1}/{len(indices)}", flush=True)

    return {
        "frame_counts": dict(frame_counts),
        "frame_counts_note": "global per-channel thresholds; startup exclude applied",
        "truncated_episodes": truncated_eps,
        "prefix_attribution_first_abnormal": dict(prefix_attr),
        "position_histograms": {k: v.tolist() for k, v in pos_hist.items()},
    }


def save_plots(out: Path, stage1: dict, log_summary: dict) -> None:
    centers = np.arange(0.5, 100, 1.0)
    pos_hist = {k: np.array(v) for k, v in stage1["position_histograms"].items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        ("residual", "Residual > global threshold", "#2563eb"),
        ("accel", "Accel > global threshold", "#16a34a"),
        ("jerk", "Jerk > global threshold", "#dc2626"),
        ("sudden", "Sudden: residual & (accel|jerk)", "#7c3aed"),
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
    n = log_summary["total_episodes"]
    fig.suptitle(f"Stage1 global thresholds · position distribution · {n} episodes", fontsize=13)
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
    ax.set_title("Sudden-change frames by triggering component (global thresholds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "stage1_sudden_position_by_component.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    attr = stage1["prefix_attribution_first_abnormal"]
    labels = list(attr.keys())
    vals = [attr[k] for k in labels]
    ax.bar(labels, vals, color="#6366f1", edgecolor="none")
    ax.set_ylabel("Episodes")
    ax.set_title(
        f"First abnormal frame cause · truncated: {stage1['truncated_episodes']}/{n}"
    )
    plt.xticks(rotation=25, ha="right")
    for idx, v in enumerate(vals):
        ax.text(idx, v + max(vals) * 0.01 + 1, str(v), ha="center", fontsize=9)
    fig.savefig(out / "prefix_attribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = ["Fully kept", "Prefix truncated"]
    vals = [log_summary["fully_kept_episodes"], log_summary["truncated_episodes"]]
    ax.bar(labels, vals, color=["#22c55e", "#f97316"])
    ax.set_ylabel("Episodes")
    ax.set_title("Episode-level truncation summary")
    for idx, v in enumerate(vals):
        ax.text(idx, v + 20, f"{v} ({100*v/n:.1f}%)", ha="center")
    fig.savefig(out / "truncation_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze pipeline experiment results")
    parser.add_argument("--exp-dir", type=Path, default=Path("output/exp3000_v10"))
    parser.add_argument("--episode-limit", type=int, default=None)
    args = parser.parse_args()

    exp = args.exp_dir
    log_path = exp / "reports" / "exclusion_log.jsonl"
    quality_path = exp / "reports" / "quality_report.json"
    s3_cache = exp / "cache" / "global_stats.npz"
    s1_cache = exp / "cache" / "stage1_global_stats.npz"
    out = exp / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_config_from_yaml(load_config(ROOT / "config" / "humanoid_merged.yaml"), {})
    n = args.episode_limit or json.loads(quality_path.read_text())["summary"]["total_episodes"]
    indices = list_episode_indices(pipe.dataset_root, n)[:n]

    print("Analyzing exclusion log...")
    log_summary = analyze_exclusion_log(log_path)

    print("Analyzing Stage1 with global thresholds...")
    s3_stats = GlobalStats.load(str(s3_cache))
    s1_stats = Stage1GlobalStats.load(str(s1_cache))
    stage1 = analyze_stage1_global(pipe, indices, s3_stats, s1_stats)

    report = {
        "experiment_dir": str(exp),
        "quality_report_summary": json.loads(quality_path.read_text())["summary"],
        "exclusion_log_summary": log_summary,
        "stage1_global_analysis": {k: v for k, v in stage1.items() if k != "position_histograms"},
    }
    report_path = out / "analysis_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (out / "stage1_position_histograms.json").open("w", encoding="utf-8") as f:
        json.dump(stage1["position_histograms"], f)

    save_plots(out, stage1, log_summary)
    print(json.dumps(report, indent=2))
    print(f"\nAnalysis written to {out}")


if __name__ == "__main__":
    main()
