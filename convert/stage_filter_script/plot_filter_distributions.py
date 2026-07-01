from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .filter_config import (
        add_config_arg,
        add_filter_override_args,
        apply_filter_overrides,
        episode_filter_config_from_dict,
        filter_keys_from_dict,
        load_filter_config,
    )
    from .filter_core import evaluate_episode, extreme_value_bounds, hand_pair_confidence
    from .filter_lerobot_dataset import array_column, data_file_path, load_json, load_jsonl, normalize_feature_shapes
except ImportError:
    from filter_config import (
        add_config_arg,
        add_filter_override_args,
        apply_filter_overrides,
        episode_filter_config_from_dict,
        filter_keys_from_dict,
        load_filter_config,
    )
    from filter_core import evaluate_episode, extreme_value_bounds, hand_pair_confidence
    from filter_lerobot_dataset import array_column, data_file_path, load_json, load_jsonl, normalize_feature_shapes


DEFAULT_INPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_lerobot_v21")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/filter_distributions/egodex_demo_lerobot_v21")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def histogram(values: np.ndarray, bins: int, value_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = finite_values(values)
    if len(values) == 0:
        return np.zeros(bins, dtype=np.int64), np.linspace(0.0, 1.0, bins + 1)
    if value_range is None:
        lo, hi = np.nanpercentile(values, [0.5, 99.5])
        if abs(hi - lo) < 1e-12:
            lo -= 0.5
            hi += 0.5
        value_range = (float(lo), float(hi))
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    return counts.astype(np.int64), edges.astype(np.float64)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_hist_svg(
    path: Path,
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    threshold: float | None = None,
    bins: int = 60,
    value_range: tuple[float, float] | None = None,
) -> None:
    counts, edges = histogram(values, bins=bins, value_range=value_range)
    width, height = 920, 520
    ml, mr, mt, mb = 76, 28, 56, 78
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_count = max(int(counts.max()), 1)
    x_min, x_max = float(edges[0]), float(edges[-1])

    def x_pos(x: float) -> float:
        if abs(x_max - x_min) < 1e-12:
            return ml + plot_w / 2
        return ml + (x - x_min) / (x_max - x_min) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfb"/>',
        f'<text x="{ml}" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#222">{svg_escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt + plot_h}" x2="{ml + plot_w}" y2="{mt + plot_h}" stroke="#444" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}" stroke="#444" stroke-width="1"/>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = mt + plot_h - frac * plot_h
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{int(frac * max_count)}</text>')

    bar_gap = 1.0
    for i, count in enumerate(counts):
        x0 = x_pos(float(edges[i]))
        x1 = x_pos(float(edges[i + 1]))
        bar_w = max(1.0, x1 - x0 - bar_gap)
        bar_h = (count / max_count) * plot_h
        y = mt + plot_h - bar_h
        parts.append(f'<rect x="{x0:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#3d74b8" opacity="0.88"/>')

    if threshold is not None and x_min <= threshold <= x_max:
        x = x_pos(threshold)
        parts.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{mt + plot_h}" stroke="#d23f31" stroke-width="2" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{x + 6:.2f}" y="{mt + 18}" font-family="Arial" font-size="13" fill="#d23f31">threshold={threshold:.4g}</text>')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = ml + frac * plot_w
        value = x_min + frac * (x_max - x_min)
        parts.append(f'<text x="{x:.1f}" y="{mt + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">{value:.4g}</text>')

    parts.append(f'<text x="{ml + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="15" fill="#333">{svg_escape(xlabel)}</text>')
    parts.append(f'<text x="18" y="{mt + plot_h / 2}" transform="rotate(-90 18 {mt + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#333">count</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_svg(
    path: Path,
    labels: list[str],
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    threshold: float | None = None,
) -> None:
    width, height = 980, 540
    ml, mr, mt, mb = 76, 36, 56, 110
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    max_value = max(float(np.nanmax(values)) if len(values) else 0.0, threshold or 0.0, 1e-6)
    n = max(len(values), 1)
    bar_w = plot_w / n * 0.72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfb"/>',
        f'<text x="{ml}" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#222">{svg_escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt + plot_h}" x2="{ml + plot_w}" y2="{mt + plot_h}" stroke="#444" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}" stroke="#444" stroke-width="1"/>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = mt + plot_h - frac * plot_h
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{frac * max_value:.3f}</text>')

    for i, (label, value) in enumerate(zip(labels, values)):
        x_center = ml + (i + 0.5) / n * plot_w
        bar_h = max(0.0, float(value)) / max_value * plot_h
        y = mt + plot_h - bar_h
        color = "#d95f4f" if threshold is not None and value > threshold else "#4f8f69"
        parts.append(f'<rect x="{x_center - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}" opacity="0.88"/>')
        if n <= 40:
            parts.append(f'<text x="{x_center:.1f}" y="{mt + plot_h + 18}" text-anchor="end" transform="rotate(-55 {x_center:.1f} {mt + plot_h + 18})" font-family="Arial" font-size="11" fill="#555">{svg_escape(label)}</text>')

    if threshold is not None:
        y = mt + plot_h - threshold / max_value * plot_h
        parts.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + plot_w}" y2="{y:.2f}" stroke="#d23f31" stroke-width="2" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{ml + plot_w - 4}" y="{y - 6:.2f}" text-anchor="end" font-family="Arial" font-size="13" fill="#d23f31">threshold={threshold:.3f}</text>')

    parts.append(f'<text x="18" y="{mt + plot_h / 2}" transform="rotate(-90 18 {mt + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#333">{svg_escape(ylabel)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_multi_hist_svg(
    path: Path,
    series: list[np.ndarray],
    *,
    titles: list[str],
    page_title: str,
    xlabel: str,
    thresholds: list[tuple[float | None, float | None]] | None = None,
    bins: int = 36,
    cols: int = 4,
) -> None:
    rows = int(np.ceil(len(series) / cols))
    cell_w, cell_h = 260, 190
    width = cols * cell_w + 40
    height = rows * cell_h + 76
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfb"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#222">{svg_escape(page_title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">{svg_escape(xlabel)}</text>',
    ]

    for i, values in enumerate(series):
        r, c = divmod(i, cols)
        x0 = 24 + c * cell_w
        y0 = 56 + r * cell_h
        plot_w = cell_w - 42
        plot_h = cell_h - 58
        vals = finite_values(values)
        if len(vals) == 0:
            vals = np.array([0.0])
        lo, hi = np.nanpercentile(vals, [0.5, 99.5])
        threshold_pair = thresholds[i] if thresholds else (None, None)
        finite_thresholds = [t for t in threshold_pair if t is not None and np.isfinite(t)]
        if finite_thresholds:
            lo = min(float(lo), min(finite_thresholds))
            hi = max(float(hi), max(finite_thresholds))
        if abs(hi - lo) < 1e-12:
            lo -= 0.5
            hi += 0.5
        counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
        max_count = max(int(counts.max()), 1)

        parts.append(f'<text x="{x0}" y="{y0}" font-family="Arial" font-size="13" font-weight="700" fill="#222">{svg_escape(titles[i])}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#555" stroke-width="1"/>')
        parts.append(f'<line x1="{x0}" y1="{y0 + 18}" x2="{x0}" y2="{y0 + plot_h}" stroke="#555" stroke-width="1"/>')

        def x_pos(x: float) -> float:
            return x0 + (x - lo) / (hi - lo) * plot_w

        for j, count in enumerate(counts):
            bx0 = x_pos(float(edges[j]))
            bx1 = x_pos(float(edges[j + 1]))
            bh = count / max_count * (plot_h - 18)
            by = y0 + plot_h - bh
            parts.append(f'<rect x="{bx0:.2f}" y="{by:.2f}" width="{max(1.0, bx1 - bx0 - 0.6):.2f}" height="{bh:.2f}" fill="#3d74b8" opacity="0.84"/>')

        for threshold in finite_thresholds:
            if lo <= threshold <= hi:
                tx = x_pos(float(threshold))
                parts.append(f'<line x1="{tx:.2f}" y1="{y0 + 18}" x2="{tx:.2f}" y2="{y0 + plot_h}" stroke="#d23f31" stroke-width="1.5" stroke-dasharray="4 4"/>')

        parts.append(f'<text x="{x0}" y="{y0 + plot_h + 16}" font-family="Arial" font-size="10" fill="#555">{lo:.3g}</text>')
        parts.append(f'<text x="{x0 + plot_w}" y="{y0 + plot_h + 16}" text-anchor="end" font-family="Arial" font-size="10" fill="#555">{hi:.3g}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    reset_dir(output_dir, args.overwrite)

    info = normalize_feature_shapes(load_json(dataset_root / "meta/info.json"))
    episodes = load_jsonl(dataset_root / "meta/episodes.jsonl")
    raw_config = apply_filter_overrides(load_filter_config(args.config), args, include_stage2=False)
    raw_config.setdefault("stage2", {})["enabled"] = False
    config = episode_filter_config_from_dict(raw_config)
    keys = filter_keys_from_dict(raw_config)
    if keys["confidence_key"] in {"", "none", "None", "null"}:
        keys["confidence_key"] = None

    actions: list[np.ndarray] = []
    dfs: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for ep in episodes:
        df = pd.read_parquet(dataset_root / data_file_path(info, ep["episode_index"]))
        action = array_column(df, keys["action_key"])
        if action is None:
            raise KeyError(f"{keys['action_key']} not found in episode {ep['episode_index']}")
        actions.append(action)
        dfs.append((ep, df))
    all_action = np.concatenate(actions, axis=0)
    stage3_bounds = extreme_value_bounds(all_action, config.stage3)

    residual_values_by_dim: list[list[np.ndarray]] = [[] for _ in range(all_action.shape[1])]
    accel_values_by_dim: list[list[np.ndarray]] = [[] for _ in range(all_action.shape[1])]
    jerk_values_by_dim: list[list[np.ndarray]] = [[] for _ in range(all_action.shape[1])]
    action_values_by_dim: list[list[np.ndarray]] = [[] for _ in range(all_action.shape[1])]
    residual_thresholds_by_ep = []
    stage3_violation_values = []
    confidence_scores = []
    episode_rows = []
    dim_stage1_counts = np.zeros(all_action.shape[1], dtype=np.int64)
    dim_stage3_counts = np.zeros(all_action.shape[1], dtype=np.int64)

    for ep, df in dfs:
        action = array_column(df, keys["action_key"])
        confidence = array_column(df, keys["confidence_key"]) if keys["confidence_key"] else None
        result = evaluate_episode(action, confidence=confidence, config=config, stage3_bounds=stage3_bounds)
        stage1 = result["stage1"]
        stage3 = result["stage3"]

        residual_thresholds_by_ep.append(stage1["residual_threshold"])

        lower, upper = stage3_bounds
        lower_violation = np.zeros_like(action, dtype=np.float32)
        upper_violation = np.zeros_like(action, dtype=np.float32)
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        lower_violation[:, finite_lower] = (
            lower[finite_lower] - action[:, finite_lower]
        ) / np.maximum(np.abs(lower[finite_lower]), 1e-6)
        upper_violation[:, finite_upper] = (
            action[:, finite_upper] - upper[finite_upper]
        ) / np.maximum(np.abs(upper[finite_upper]), 1e-6)
        stage3_score = np.max(np.maximum(0.0, np.maximum(lower_violation, upper_violation)), axis=1)

        stage3_violation_values.append(stage3_score)
        dim_stage1_counts += np.sum(stage1["dim_mask"], axis=0)
        dim_stage3_counts += np.sum(stage3["dim_mask"], axis=0)
        for dim in range(action.shape[1]):
            residual_values_by_dim[dim].append(stage1["residual"][:, dim])
            accel_values_by_dim[dim].append(stage1["accel"][:, dim])
            jerk_values_by_dim[dim].append(stage1["jerk"][:, dim])
            action_values_by_dim[dim].append(action[:, dim])

        if confidence is not None:
            conf_score = hand_pair_confidence(confidence)
            confidence_scores.append(conf_score)
            confidence_bad_ratio = float(np.mean(conf_score < config.confidence_min)) if config.confidence_min is not None else 0.0
        else:
            confidence_bad_ratio = 0.0

        episode_rows.append(
            {
                "episode_index": ep["episode_index"],
                "length": len(action),
                "keep_episode": result["keep_episode"],
                "bad_frame_ratio": result["bad_frame_ratio"],
                "stage1_bad_ratio": float(np.mean(stage1["frame_mask"])),
                "stage3_bad_ratio": float(np.mean(stage3["frame_mask"])),
                "confidence_bad_ratio": confidence_bad_ratio,
                "reasons": ";".join(result["reasons"]),
            }
        )

    stage3_values = np.concatenate(stage3_violation_values)
    confidence_values = np.concatenate(confidence_scores) if confidence_scores else np.array([], dtype=np.float32)

    residual_true_by_dim = [np.concatenate(items) for items in residual_values_by_dim]
    accel_true_by_dim = [np.concatenate(items) for items in accel_values_by_dim]
    jerk_true_by_dim = [np.concatenate(items) for items in jerk_values_by_dim]
    action_true_by_dim = [np.concatenate(items) for items in action_values_by_dim]
    residual_thresholds = np.nanmedian(np.stack(residual_thresholds_by_ep), axis=0)
    dim_titles = [f"dim {i}" for i in range(all_action.shape[1])]

    write_multi_hist_svg(
        output_dir / "stage1_residual_true_by_dim.svg",
        residual_true_by_dim,
        titles=dim_titles,
        page_title="Stage 1 residual true-value distribution by action dimension",
        xlabel="residual = |action - trend|",
        thresholds=[(None, float(t)) for t in residual_thresholds],
    )
    write_multi_hist_svg(
        output_dir / "stage1_accel_true_by_dim.svg",
        accel_true_by_dim,
        titles=dim_titles,
        page_title="Acceleration true-value distribution by action dimension (diagnostic only)",
        xlabel="abs(second difference of action)",
    )
    write_multi_hist_svg(
        output_dir / "stage1_jerk_true_by_dim.svg",
        jerk_true_by_dim,
        titles=dim_titles,
        page_title="Jerk true-value distribution by action dimension (diagnostic only)",
        xlabel="abs(third difference of action)",
    )
    write_multi_hist_svg(
        output_dir / "stage3_action_true_by_dim.svg",
        action_true_by_dim,
        titles=dim_titles,
        page_title="Stage 3 action true-value distribution by action dimension",
        xlabel="action value",
        thresholds=[(float(stage3_bounds[0][i]), float(stage3_bounds[1][i])) for i in range(all_action.shape[1])],
    )
    write_hist_svg(output_dir / "stage3_violation_score_hist.svg", stage3_values, title="Stage 3 extreme-value violation distribution", xlabel="relative distance outside global bounds", threshold=0.0, value_range=(0.0, max(float(np.nanpercentile(stage3_values, 99.5)), 1e-3)))
    if len(confidence_values):
        write_hist_svg(output_dir / "confidence_min_hist.svg", confidence_values, title="Per-frame hand-pair confidence distribution", xlabel="min of wrist*fingertip scores across both hands", threshold=config.confidence_min, value_range=(0.0, 1.0))

    ep_labels = [str(row["episode_index"]) for row in episode_rows]
    write_bar_svg(
        output_dir / "episode_bad_frame_ratio.svg",
        ep_labels,
        np.array([row["bad_frame_ratio"] for row in episode_rows], dtype=np.float32),
        title="Episode bad-frame ratio",
        ylabel="bad frames / frames",
        threshold=config.max_bad_frame_ratio,
    )
    write_bar_svg(
        output_dir / "stage1_dim_flag_counts.svg",
        [str(i) for i in range(len(dim_stage1_counts))],
        dim_stage1_counts.astype(np.float32),
        title="Stage 1 flagged count by action dimension",
        ylabel="flagged dimension count",
    )
    write_bar_svg(
        output_dir / "stage3_dim_flag_counts.svg",
        [str(i) for i in range(len(dim_stage3_counts))],
        dim_stage3_counts.astype(np.float32),
        title="Stage 3 flagged count by action dimension",
        ylabel="flagged dimension count",
    )

    write_csv(output_dir / "episode_filter_distribution.csv", episode_rows)
    threshold_rows = []
    for dim in range(all_action.shape[1]):
        threshold_rows.append(
            {
                "action_dim": dim,
                "stage1_residual_threshold": float(residual_thresholds[dim]),
                "stage1_residual_threshold_rule": "mean(abs(residual)) * residual_mean_multiplier",
                "stage3_lower": float(stage3_bounds[0][dim]),
                "stage3_upper": float(stage3_bounds[1][dim]),
            }
        )
    write_csv(output_dir / "filter_thresholds_by_dim.csv", threshold_rows)
    write_json(
        output_dir / "distribution_summary.json",
        {
            "dataset": dataset_root,
            "output_dir": output_dir,
            "num_episodes": len(episodes),
            "num_frames": int(sum(row["length"] for row in episode_rows)),
            "kept_episodes_at_current_thresholds": int(sum(bool(row["keep_episode"]) for row in episode_rows)),
            "config": config,
            "config_path": str(args.config),
            "data_keys": keys,
            "stage3_bounds": stage3_bounds,
            "plots": sorted(path.name for path in output_dir.glob("*.svg")),
        },
    )
    print(f"Wrote distribution plots to {output_dir}")
    print(f"Episodes kept at current thresholds: {sum(bool(row['keep_episode']) for row in episode_rows)}/{len(episode_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot filtering metric distributions for an original LeRobot dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    add_config_arg(parser)
    add_filter_override_args(parser, include_stage2=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
