from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_lerobot_v21")
DEFAULT_OUTPUT_DIR = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/filter_failure_visualizations")
FILTER_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "filter_script"
if str(FILTER_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FILTER_SCRIPT_DIR))

from filter_config import (  # noqa: E402
    add_config_arg,
    add_filter_override_args,
    apply_filter_overrides,
    episode_filter_config_from_dict,
    filter_keys_from_dict,
    load_filter_config,
)
from filter_core import evaluate_episode, extreme_value_bounds, hand_pair_confidence  # noqa: E402
from filter_lerobot_dataset import array_column, data_file_path, load_json, load_jsonl, normalize_feature_shapes  # noqa: E402


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_bounds(values: np.ndarray, *, include_zero: bool = False) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if abs(hi - lo) < 1e-12:
        pad = max(abs(lo) * 0.05, 1e-3)
    else:
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def x_scale(frame: np.ndarray | float, x0: float, width: float, n: int) -> np.ndarray | float:
    if n <= 1:
        return x0 + width / 2
    return x0 + np.asarray(frame) / (n - 1) * width


def y_scale(values: np.ndarray, y0: float, height: float, lo: float, hi: float) -> np.ndarray:
    if abs(hi - lo) < 1e-12:
        return np.full_like(values, y0 + height / 2, dtype=np.float32)
    return y0 + height - (values - lo) / (hi - lo) * height


def polyline(values: np.ndarray, x0: float, y0: float, width: float, height: float, lo: float, hi: float) -> str:
    frames = np.arange(len(values), dtype=np.float32)
    xs = x_scale(frames, x0, width, len(values))
    ys = y_scale(values.astype(np.float32), y0, height, lo, hi)
    return " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in zip(xs, ys))


def mask_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    groups = np.split(idx, breaks)
    return [(int(group[0]), int(group[-1])) for group in groups if len(group)]


def draw_frame_ticks(parts: list[str], *, x0: float, y0: float, width: float, n: int) -> None:
    ticks = [0, max(n - 1, 0)] if n <= 2 else [0, (n - 1) // 2, n - 1]
    for frame in ticks:
        x = float(x_scale(float(frame), x0, width, n))
        anchor = "middle"
        if frame == 0:
            anchor = "start"
        elif frame == n - 1:
            anchor = "end"
        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + 5}" stroke="#999"/>')
        parts.append(f'<text x="{x:.2f}" y="{y0 + 18}" text-anchor="{anchor}" font-family="Arial" font-size="10" fill="#666">{frame}</text>')


def draw_y_labels(parts: list[str], *, x0: float, y0: float, height: float, lo: float, hi: float) -> None:
    parts.append(f'<text x="{x0 - 6}" y="{y0 + 4}" text-anchor="end" font-family="Arial" font-size="9" fill="#777">{hi:.3g}</text>')
    parts.append(f'<text x="{x0 - 6}" y="{y0 + height:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="#777">{lo:.3g}</text>')


def draw_grid(parts: list[str], *, x0: float, y0: float, width: float, height: float) -> None:
    for frac in [0.25, 0.5, 0.75]:
        y = y0 + height * frac
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#eeeeee"/>')


def draw_bad_frame_background(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    n: int,
    mask: np.ndarray,
    fill: str,
) -> None:
    for start, end in mask_spans(mask):
        x1 = float(x_scale(float(start), x0, width, n))
        x2 = float(x_scale(float(end), x0, width, n))
        parts.append(f'<rect x="{x1:.2f}" y="{y0}" width="{max(2.0, x2 - x1 + 2.0):.2f}" height="{height}" fill="{fill}"/>')


def draw_stage_strip(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    n: int,
    stage1: np.ndarray,
    stage3: np.ndarray,
    confidence: np.ndarray,
    combined: np.ndarray,
) -> None:
    rows = [
        ("Stage1 residual", stage1, "#d64235", "rgba(214,66,53,0.22)"),
        ("Stage3 extreme", stage3, "#7b4cc2", "rgba(123,76,194,0.22)"),
        ("Low confidence", confidence, "#d99023", "rgba(217,144,35,0.24)"),
        ("Any bad frame", combined, "#111111", "rgba(17,17,17,0.18)"),
    ]
    row_h = 22
    gap = 8
    parts.append(f'<text x="{x0}" y="{y0 - 10}" font-family="Arial" font-size="14" font-weight="700" fill="#222">Bad-frame timeline</text>')
    for row_idx, (label, mask, color, fill) in enumerate(rows):
        y = y0 + row_idx * (row_h + gap)
        parts.append(f'<text x="{x0}" y="{y + 13}" font-family="Arial" font-size="12" fill="#333">{svg_escape(label)}</text>')
        bx0 = x0 + 130
        parts.append(f'<rect x="{bx0}" y="{y}" width="{width}" height="{row_h}" rx="2" fill="#eeeeee" stroke="#cccccc"/>')
        for start, end in mask_spans(mask):
            x1 = float(x_scale(float(start), bx0, width, n))
            x2 = float(x_scale(float(end), bx0, width, n))
            span_w = max(2.0, x2 - x1 + 2.0)
            parts.append(f'<rect x="{x1:.2f}" y="{y + 1}" width="{span_w:.2f}" height="{row_h - 2}" fill="{fill}"/>')
            parts.append(f'<line x1="{x1:.2f}" y1="{y}" x2="{x1:.2f}" y2="{y + row_h}" stroke="{color}" stroke-width="1.6"/>')
            if end != start:
                parts.append(f'<line x1="{x2:.2f}" y1="{y}" x2="{x2:.2f}" y2="{y + row_h}" stroke="{color}" stroke-width="1.2" opacity="0.75"/>')
        count = int(np.sum(mask))
        parts.append(f'<text x="{bx0 + width + 8}" y="{y + 14}" font-family="Arial" font-size="11" fill="#555">{count}</text>')
    draw_frame_ticks(parts, x0=x0 + 130, y0=y0 + len(rows) * (row_h + gap) - gap + 2, width=width, n=n)


def draw_confidence_panel(
    parts: list[str],
    *,
    confidence_score: np.ndarray | None,
    threshold: float | None,
    low_confidence_mask: np.ndarray | None,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> None:
    if confidence_score is None:
        parts.append(f'<text x="{x0}" y="{y0 + 18}" font-family="Arial" font-size="13" fill="#555">No confidence key configured.</text>')
        return
    n = len(confidence_score)
    lo, hi = 0.0, 1.0
    pts = polyline(confidence_score, x0, y0, width, height, lo, hi)
    parts.append(f'<text x="{x0}" y="{y0 - 10}" font-family="Arial" font-size="14" font-weight="700" fill="#222">Hand-pair confidence per frame</text>')
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#d0d0d0"/>')
    if low_confidence_mask is not None:
        for start, end in mask_spans(low_confidence_mask):
            x1 = float(x_scale(float(start), x0, width, n))
            x2 = float(x_scale(float(end), x0, width, n))
            parts.append(f'<rect x="{x1:.2f}" y="{y0}" width="{max(2.0, x2 - x1 + 2.0):.2f}" height="{height}" fill="rgba(217,144,35,0.16)"/>')
    for frac in [0.0, 0.5, 1.0]:
        y = y0 + height - frac * height
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#666">{frac:.1f}</text>')
    if threshold is not None:
        y = y0 + height - threshold * height
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#d99023" stroke-width="2" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x0 + width - 4}" y="{y - 5:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#b36b13">threshold={threshold:.3g}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#2f70b7" stroke-width="1.8"/>')
    draw_frame_ticks(parts, x0=x0, y0=y0 + height, width=width, n=n)


def draw_dim_panel(
    parts: list[str],
    *,
    dim: int,
    name: str,
    action: np.ndarray,
    trend: np.ndarray,
    residual: np.ndarray,
    residual_threshold: float,
    stage1_mask: np.ndarray,
    stage3_mask: np.ndarray,
    stage3_lower: float,
    stage3_upper: float,
    x0: float,
    y0: float,
    width: float,
    action_h: float,
    residual_h: float,
) -> None:
    n = len(action)
    action_lo, action_hi = nice_bounds(np.concatenate([action, trend]), include_zero=False)
    if np.isfinite(stage3_lower):
        action_lo = min(action_lo, stage3_lower)
        action_hi = max(action_hi, stage3_lower)
    if np.isfinite(stage3_upper):
        action_lo = min(action_lo, stage3_upper)
        action_hi = max(action_hi, stage3_upper)
    residual_lo, residual_hi = nice_bounds(np.concatenate([residual, np.array([residual_threshold])]), include_zero=True)

    combined_mask = stage1_mask | stage3_mask

    parts.append(f'<text x="{x0}" y="{y0 - 10}" font-family="Arial" font-size="14" font-weight="700" fill="#222">dim {dim}: {svg_escape(name)}</text>')
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{action_h}" fill="#ffffff" stroke="#d0d0d0"/>')
    draw_grid(parts, x0=x0, y0=y0, width=width, height=action_h)
    draw_y_labels(parts, x0=x0, y0=y0, height=action_h, lo=action_lo, hi=action_hi)
    draw_bad_frame_background(
        parts,
        x0=x0,
        y0=y0,
        width=width,
        height=action_h,
        n=n,
        mask=combined_mask,
        fill="rgba(17,17,17,0.06)",
    )
    raw_pts = polyline(action, x0, y0, width, action_h, action_lo, action_hi)
    trend_pts = polyline(trend, x0, y0, width, action_h, action_lo, action_hi)
    parts.append(f'<polyline points="{raw_pts}" fill="none" stroke="#c84e40" stroke-width="1.4" opacity="0.8"/>')
    parts.append(f'<polyline points="{trend_pts}" fill="none" stroke="#2f70b7" stroke-width="1.8"/>')
    for value, label in [(stage3_lower, "lower"), (stage3_upper, "upper")]:
        if np.isfinite(value) and action_lo <= value <= action_hi:
            y = y_scale(np.array([value], dtype=np.float32), y0, action_h, action_lo, action_hi)[0]
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#7b4cc2" stroke-width="1.4" stroke-dasharray="5 4"/>')
            parts.append(f'<text x="{x0 + width - 4}" y="{y - 4:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="#7b4cc2">{label}</text>')

    stage3_idx = np.flatnonzero(stage3_mask)
    if len(stage3_idx):
        xs = x_scale(stage3_idx.astype(np.float32), x0, width, n)
        ys = y_scale(action[stage3_idx], y0, action_h, action_lo, action_hi)
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{float(x):.2f}" cy="{float(y):.2f}" r="3.2" fill="#7b4cc2" stroke="#ffffff" stroke-width="0.8"/>')

    ry0 = y0 + action_h + 20
    parts.append(f'<rect x="{x0}" y="{ry0}" width="{width}" height="{residual_h}" fill="#ffffff" stroke="#d0d0d0"/>')
    draw_grid(parts, x0=x0, y0=ry0, width=width, height=residual_h)
    draw_y_labels(parts, x0=x0, y0=ry0, height=residual_h, lo=residual_lo, hi=residual_hi)
    draw_bad_frame_background(
        parts,
        x0=x0,
        y0=ry0,
        width=width,
        height=residual_h,
        n=n,
        mask=stage1_mask,
        fill="rgba(214,66,53,0.08)",
    )
    residual_pts = polyline(residual, x0, ry0, width, residual_h, residual_lo, residual_hi)
    parts.append(f'<polyline points="{residual_pts}" fill="none" stroke="#444444" stroke-width="1.5"/>')
    threshold_y = y_scale(np.array([residual_threshold], dtype=np.float32), ry0, residual_h, residual_lo, residual_hi)[0]
    parts.append(f'<line x1="{x0}" y1="{threshold_y:.1f}" x2="{x0 + width}" y2="{threshold_y:.1f}" stroke="#d64235" stroke-width="1.6" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{x0 + width - 4}" y="{threshold_y - 5:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="#d64235">residual threshold={residual_threshold:.3g}</text>')
    stage1_idx = np.flatnonzero(stage1_mask)
    if len(stage1_idx):
        xs = x_scale(stage1_idx.astype(np.float32), x0, width, n)
        ys = y_scale(residual[stage1_idx], ry0, residual_h, residual_lo, residual_hi)
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{float(x):.2f}" cy="{float(y):.2f}" r="3.2" fill="#d64235" stroke="#ffffff" stroke-width="0.8"/>')

    draw_frame_ticks(parts, x0=x0, y0=ry0 + residual_h + 3, width=width, n=n)
    parts.append(f'<text x="{x0}" y="{ry0 + residual_h + 32}" font-family="Arial" font-size="10" fill="#666">gray bands: bad frames for this dimension; red dots: residual threshold; purple dots: action bounds</text>')


def collect_stage3_bounds(dataset_root: Path, info: dict[str, Any], episodes: list[dict[str, Any]], action_key: str, config) -> tuple[np.ndarray, np.ndarray]:
    actions = []
    for ep in episodes:
        df = pd.read_parquet(dataset_root / data_file_path(info, ep["episode_index"]))
        actions.append(array_column(df, action_key))
    return extreme_value_bounds(np.concatenate(actions, axis=0), config.stage3)


def visualize_episode(args: argparse.Namespace, ep_index: int) -> Path:
    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = normalize_feature_shapes(load_json(dataset_root / "meta/info.json"))
    episodes = load_jsonl(dataset_root / "meta/episodes.jsonl")
    raw_config = apply_filter_overrides(load_filter_config(args.config), args, include_stage2=True)
    config = episode_filter_config_from_dict(raw_config)
    keys = filter_keys_from_dict(raw_config)
    if keys["confidence_key"] in {"", "none", "None", "null"}:
        keys["confidence_key"] = None
    if keys["state_key"] in {"", "none", "None", "null"}:
        keys["state_key"] = None

    stage3_bounds = collect_stage3_bounds(dataset_root, info, episodes, keys["action_key"], config)
    df = pd.read_parquet(dataset_root / data_file_path(info, ep_index))
    action = array_column(df, keys["action_key"])
    confidence = array_column(df, keys["confidence_key"]) if keys["confidence_key"] else None
    state = array_column(df, keys["state_key"]) if keys["state_key"] else None
    result = evaluate_episode(action, confidence=confidence, state=state, config=config, stage3_bounds=stage3_bounds)

    names = info["features"][keys["action_key"]].get("names") or [f"action_{i}" for i in range(action.shape[1])]
    n = len(action)
    confidence_score = hand_pair_confidence(confidence)
    confidence_bad = result["confidence_bad_mask"]

    flagged_dims = np.flatnonzero(np.any(result["stage1"]["dim_mask"] | result["stage3"]["dim_mask"], axis=0))
    if args.only_flagged_dims and len(flagged_dims):
        dims = flagged_dims.tolist()
    else:
        dims = list(range(action.shape[1]))
    if args.max_dims is not None:
        dims = dims[: args.max_dims]

    panel_w = 560
    action_h = 130
    residual_h = 92
    panel_h = action_h + residual_h + 92
    cols = max(args.cols, 1)
    rows = int(np.ceil(max(len(dims), 1) / cols))
    left = 64
    right = 40
    panel_gap = 42
    width = left + right + cols * panel_w + (cols - 1) * panel_gap
    plot_w = width - left - right
    strip_y = 132
    strip_w = plot_w - 130
    confidence_y = strip_y + 158
    confidence_h = 96
    confidence_w = plot_w
    header_h = confidence_y + confidence_h + 64
    height = header_h + rows * panel_h + 40

    reasons = ", ".join(result["reasons"]) if result["reasons"] else "none"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfb"/>',
        f'<text x="{left}" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#222">Filter diagnosis: episode {ep_index:06d}</text>',
        f'<text x="{left}" y="58" font-family="Arial" font-size="13" fill="#333">keep_episode={result["keep_episode"]}; bad_frame_ratio={result["bad_frame_ratio"]:.3f}; reasons={svg_escape(reasons)}</text>',
        f'<text x="{left}" y="78" font-family="Arial" font-size="12" fill="#555">Stage1: residual &gt; mean(abs(residual)) * {config.stage1.residual_mean_multiplier}; Stage3: action outside global bounds; confidence threshold={config.confidence_min}</text>',
        f'<line x1="{left}" y1="102" x2="{left + 30}" y2="102" stroke="#c84e40" stroke-width="2"/>',
        f'<text x="{left + 36}" y="106" font-family="Arial" font-size="12" fill="#333">raw action</text>',
        f'<line x1="{left + 126}" y1="102" x2="{left + 156}" y2="102" stroke="#2f70b7" stroke-width="2"/>',
        f'<text x="{left + 162}" y="106" font-family="Arial" font-size="12" fill="#333">smoothed/trend</text>',
        f'<circle cx="{left + 288}" cy="102" r="4" fill="#d64235"/>',
        f'<text x="{left + 298}" y="106" font-family="Arial" font-size="12" fill="#333">Stage1 threshold point</text>',
        f'<circle cx="{left + 470}" cy="102" r="4" fill="#7b4cc2"/>',
        f'<text x="{left + 480}" y="106" font-family="Arial" font-size="12" fill="#333">Stage3 threshold point</text>',
    ]

    draw_stage_strip(
        parts,
        x0=left,
        y0=strip_y,
        width=strip_w,
        n=n,
        stage1=result["stage1"]["frame_mask"],
        stage3=result["stage3"]["frame_mask"],
        confidence=confidence_bad,
        combined=result["bad_mask"],
    )
    draw_confidence_panel(
        parts,
        confidence_score=confidence_score,
        threshold=config.confidence_min,
        low_confidence_mask=confidence_bad,
        x0=left,
        y0=confidence_y,
        width=confidence_w,
        height=confidence_h,
    )

    for idx, dim in enumerate(dims):
        row, col = divmod(idx, cols)
        x0 = left + col * (panel_w + panel_gap)
        y0 = header_h + row * panel_h
        draw_dim_panel(
            parts,
            dim=dim,
            name=names[dim],
            action=action[:, dim],
            trend=result["stage1"]["trend"][:, dim],
            residual=result["stage1"]["residual"][:, dim],
            residual_threshold=float(result["stage1"]["residual_threshold"][dim]),
            stage1_mask=result["stage1"]["dim_mask"][:, dim],
            stage3_mask=result["stage3"]["dim_mask"][:, dim],
            stage3_lower=float(result["stage3"]["lower"][dim]),
            stage3_upper=float(result["stage3"]["upper"][dim]),
            x0=x0,
            y0=y0,
            width=panel_w,
            action_h=action_h,
            residual_h=residual_h,
        )

    parts.append("</svg>")
    output_path = output_dir / f"episode_{ep_index:06d}_filter_diagnosis.svg"
    output_path.write_text("\n".join(parts), encoding="utf-8")

    summary_path = output_dir / f"episode_{ep_index:06d}_filter_diagnosis.json"
    summary = {
        "episode_index": ep_index,
        "keep_episode": result["keep_episode"],
        "bad_frame_ratio": result["bad_frame_ratio"],
        "reasons": result["reasons"],
        "stage1_bad_frames": int(np.sum(result["stage1"]["frame_mask"])),
        "stage3_bad_frames": int(np.sum(result["stage3"]["frame_mask"])),
        "confidence_bad_frames": int(np.sum(confidence_bad)),
        "flagged_dims": flagged_dims.tolist(),
        "output_svg": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize why a LeRobot episode is filtered out.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--max-dims", type=int, default=None)
    parser.add_argument("--only-flagged-dims", action="store_true")
    add_config_arg(parser)
    add_filter_override_args(parser, include_stage2=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = visualize_episode(args, args.episode_index)
    print(f"Wrote {path}")
