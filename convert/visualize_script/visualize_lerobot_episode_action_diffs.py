from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_lerobot_v21")
DEFAULT_OUTPUT_DIR = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/episode_action_diffs")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_chunk(ep_index: int, chunks_size: int) -> int:
    return ep_index // chunks_size


def data_file_path(info: dict[str, Any], ep_index: int) -> Path:
    return Path(
        info["data_path"].format(
            episode_chunk=episode_chunk(ep_index, info["chunks_size"]),
            episode_index=ep_index,
        )
    )


def array_column(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"{key!r} not found in parquet columns: {list(df.columns)}")
    values = df[key].to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values).astype(np.float32)
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist(), dtype=np.float32)
    return values.astype(np.float32)[:, None]


def odd_window(window: int, length: int) -> int:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else max(1, length - 1)
    return window


def median_filter_np(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] <= 2:
        return x.astype(np.float32)
    window = odd_window(window, x.shape[0])
    if window <= 1:
        return x.astype(np.float32)
    half = window // 2
    padded = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        out[t] = np.median(padded[t : t + window], axis=0)
    return out.astype(np.float32)


def savgol_filter_np(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] <= polyorder + 1:
        return values.astype(np.float32)
    window = odd_window(window, x.shape[0])
    if window <= polyorder + 1:
        return values.astype(np.float32)
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    vandermonde = np.vander(offsets, N=polyorder + 1, increasing=True)
    coeff = np.linalg.pinv(vandermonde)[0]
    padded = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        out[t] = np.tensordot(coeff, padded[t : t + window], axes=(0, 0))
    return out.astype(np.float32)


def smooth_for_residual(values: np.ndarray, median_window: int, savgol_window: int, polyorder: int) -> np.ndarray:
    medianed = median_filter_np(values, median_window)
    return savgol_filter_np(medianed, savgol_window, polyorder)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def polyline_points(values: np.ndarray, x0: float, y0: float, width: float, height: float, y_min: float, y_max: float) -> str:
    if len(values) == 1:
        xs = np.array([x0 + width / 2], dtype=np.float32)
    else:
        xs = x0 + np.linspace(0, width, len(values), dtype=np.float32)
    if abs(y_max - y_min) < 1e-12:
        ys = np.full(len(values), y0 + height / 2, dtype=np.float32)
    else:
        ys = y0 + height - (values - y_min) / (y_max - y_min) * height
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def nice_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return -1.0, 1.0
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if abs(y_max - y_min) < 1e-12:
        pad = max(abs(y_min) * 0.05, 1e-3)
    else:
        pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def write_small_multiples_svg(
    path: Path,
    values: np.ndarray,
    names: list[str],
    *,
    title: str,
    ylabel: str,
    start_frame: int,
    cols: int = 4,
) -> None:
    n_dims = values.shape[1]
    rows = int(np.ceil(n_dims / cols))
    cell_w, cell_h = 300, 190
    width = cols * cell_w + 44
    height = rows * cell_h + 92
    plot_w = cell_w - 64
    plot_h = cell_h - 62

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfb"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#222">{svg_escape(title)}</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="13" fill="#555">x axis: frame index, starts at {start_frame}; y axis: {svg_escape(ylabel)}</text>',
    ]

    for dim in range(n_dims):
        row, col = divmod(dim, cols)
        x0 = 24 + col * cell_w
        y0 = 80 + row * cell_h
        px0 = x0 + 48
        py0 = y0 + 26
        y_min, y_max = nice_bounds(values[:, dim])

        parts.append(f'<text x="{x0}" y="{y0 + 12}" font-family="Arial" font-size="13" font-weight="700" fill="#222">{svg_escape(names[dim])}</text>')
        for frac in [0.0, 0.5, 1.0]:
            y = py0 + plot_h - frac * plot_h
            label = y_min + frac * (y_max - y_min)
            parts.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px0 + plot_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
            parts.append(f'<text x="{px0 - 7}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#666">{label:.3g}</text>')

        if y_min < 0 < y_max:
            zero_y = py0 + plot_h - (0 - y_min) / (y_max - y_min) * plot_h
            parts.append(f'<line x1="{px0}" y1="{zero_y:.1f}" x2="{px0 + plot_w}" y2="{zero_y:.1f}" stroke="#b8b8b8" stroke-width="1" stroke-dasharray="4 4"/>')

        pts = polyline_points(values[:, dim], px0, py0, plot_w, plot_h, y_min, y_max)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#2f70b7" stroke-width="1.8"/>')
        parts.append(f'<line x1="{px0}" y1="{py0 + plot_h}" x2="{px0 + plot_w}" y2="{py0 + plot_h}" stroke="#555" stroke-width="1"/>')
        parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + plot_h}" stroke="#555" stroke-width="1"/>')
        if len(values) > 1:
            parts.append(f'<text x="{px0}" y="{py0 + plot_h + 18}" font-family="Arial" font-size="10" fill="#666">{start_frame}</text>')
            parts.append(f'<text x="{px0 + plot_w}" y="{py0 + plot_h + 18}" text-anchor="end" font-family="Arial" font-size="10" fill="#666">{start_frame + len(values) - 1}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_comparison_svg(
    path: Path,
    raw: np.ndarray,
    smooth: np.ndarray,
    names: list[str],
    *,
    title: str,
    ylabel: str,
    cols: int = 4,
) -> None:
    n_dims = raw.shape[1]
    rows = int(np.ceil(n_dims / cols))
    cell_w, cell_h = 300, 200
    width = cols * cell_w + 44
    height = rows * cell_h + 112
    plot_w = cell_w - 64
    plot_h = cell_h - 70

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfb"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#222">{svg_escape(title)}</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="13" fill="#555">x axis: frame index; y axis: {svg_escape(ylabel)}</text>',
        '<line x1="24" y1="78" x2="54" y2="78" stroke="#c84e40" stroke-width="2"/>',
        '<text x="60" y="82" font-family="Arial" font-size="13" fill="#333">raw</text>',
        '<line x1="110" y1="78" x2="140" y2="78" stroke="#2f70b7" stroke-width="2.3"/>',
        '<text x="146" y="82" font-family="Arial" font-size="13" fill="#333">smoothed</text>',
    ]

    for dim in range(n_dims):
        row, col = divmod(dim, cols)
        x0 = 24 + col * cell_w
        y0 = 100 + row * cell_h
        px0 = x0 + 48
        py0 = y0 + 26
        y_min, y_max = nice_bounds(np.concatenate([raw[:, dim], smooth[:, dim]]))

        parts.append(f'<text x="{x0}" y="{y0 + 12}" font-family="Arial" font-size="13" font-weight="700" fill="#222">{svg_escape(names[dim])}</text>')
        for frac in [0.0, 0.5, 1.0]:
            y = py0 + plot_h - frac * plot_h
            label = y_min + frac * (y_max - y_min)
            parts.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px0 + plot_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
            parts.append(f'<text x="{px0 - 7}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#666">{label:.3g}</text>')

        if y_min < 0 < y_max:
            zero_y = py0 + plot_h - (0 - y_min) / (y_max - y_min) * plot_h
            parts.append(f'<line x1="{px0}" y1="{zero_y:.1f}" x2="{px0 + plot_w}" y2="{zero_y:.1f}" stroke="#b8b8b8" stroke-width="1" stroke-dasharray="4 4"/>')

        raw_pts = polyline_points(raw[:, dim], px0, py0, plot_w, plot_h, y_min, y_max)
        smooth_pts = polyline_points(smooth[:, dim], px0, py0, plot_w, plot_h, y_min, y_max)
        parts.append(f'<polyline points="{raw_pts}" fill="none" stroke="#c84e40" stroke-width="1.5" opacity="0.78"/>')
        parts.append(f'<polyline points="{smooth_pts}" fill="none" stroke="#2f70b7" stroke-width="2.2"/>')
        parts.append(f'<line x1="{px0}" y1="{py0 + plot_h}" x2="{px0 + plot_w}" y2="{py0 + plot_h}" stroke="#555" stroke-width="1"/>')
        parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + plot_h}" stroke="#555" stroke-width="1"/>')
        parts.append(f'<text x="{px0}" y="{py0 + plot_h + 18}" font-family="Arial" font-size="10" fill="#666">0</text>')
        parts.append(f'<text x="{px0 + plot_w}" y="{py0 + plot_h + 18}" text-anchor="end" font-family="Arial" font-size="10" fill="#666">{len(raw) - 1}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_csv(path: Path, frame_start: int, values: np.ndarray, names: list[str]) -> None:
    header = ["frame_index"] + names
    rows = []
    for i, row in enumerate(values):
        rows.append(",".join([str(frame_start + i)] + [f"{float(v):.9g}" for v in row]))
    path.write_text(",".join(header) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def visualize(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_json(dataset_root / "meta/info.json")
    action_names = info["features"]["action"].get("names") or [f"action_{i}" for i in range(20)]
    ep_path = dataset_root / data_file_path(info, args.episode_index)
    if not ep_path.exists():
        raise FileNotFoundError(ep_path)

    df = pd.read_parquet(ep_path)
    action = array_column(df, args.action_key)
    if action.shape[1] != len(action_names):
        action_names = [f"action_{i}" for i in range(action.shape[1])]

    accel = np.diff(action, n=2, axis=0).astype(np.float32)
    jerk = np.diff(action, n=3, axis=0).astype(np.float32)
    smoothed = smooth_for_residual(
        action,
        median_window=args.median_window,
        savgol_window=args.savgol_window,
        polyorder=args.polyorder,
    )
    residual = np.abs(action - smoothed).astype(np.float32)

    prefix = f"episode_{args.episode_index:06d}"
    write_small_multiples_svg(
        output_dir / f"{prefix}_action_20d.svg",
        action,
        action_names,
        title=f"Episode {args.episode_index:06d} raw action, {action.shape[0]} frames",
        ylabel="raw action value",
        start_frame=0,
        cols=args.cols,
    )
    write_small_multiples_svg(
        output_dir / f"{prefix}_accel_20d.svg",
        accel,
        action_names,
        title=f"Episode {args.episode_index:06d} accel = second difference(action)",
        ylabel="second difference",
        start_frame=2,
        cols=args.cols,
    )
    write_small_multiples_svg(
        output_dir / f"{prefix}_jerk_20d.svg",
        jerk,
        action_names,
        title=f"Episode {args.episode_index:06d} jerk = third difference(action)",
        ylabel="third difference",
        start_frame=3,
        cols=args.cols,
    )
    write_small_multiples_svg(
        output_dir / f"{prefix}_residual_20d.svg",
        residual,
        action_names,
        title=(
            f"Episode {args.episode_index:06d} residual = abs(raw - smoothed), "
            f"median {args.median_window} + Savitzky-Golay {args.savgol_window}/{args.polyorder}"
        ),
        ylabel="abs(raw action - smoothed action)",
        start_frame=0,
        cols=args.cols,
    )
    write_comparison_svg(
        output_dir / f"{prefix}_raw_vs_smoothed_20d.svg",
        action,
        smoothed,
        action_names,
        title=(
            f"Episode {args.episode_index:06d} raw vs smoothed action, "
            f"median {args.median_window} + Savitzky-Golay {args.savgol_window}/{args.polyorder}"
        ),
        ylabel="action value",
        cols=args.cols,
    )

    if args.write_csv:
        save_csv(output_dir / f"{prefix}_action_20d.csv", 0, action, action_names)
        save_csv(output_dir / f"{prefix}_accel_20d.csv", 2, accel, action_names)
        save_csv(output_dir / f"{prefix}_jerk_20d.csv", 3, jerk, action_names)
        save_csv(output_dir / f"{prefix}_smoothed_20d.csv", 0, smoothed, action_names)
        save_csv(output_dir / f"{prefix}_residual_20d.csv", 0, residual, action_names)

    print(f"Wrote action/diff plots for episode {args.episode_index:06d} to {output_dir}")
    print(f"Source parquet: {ep_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one LeRobot episode's 20D action, accel, and jerk as three SVG figures.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--savgol-window", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
