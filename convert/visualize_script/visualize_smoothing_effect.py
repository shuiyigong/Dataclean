from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_NPZ = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egoverse_demo_prefilter/69a663a37f3184ccf64b0ab1.npz")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/69a663a37f3184ccf64b0ab1.png")

BG = (250, 250, 250)
PANEL_BG = (255, 255, 255)
GRID = (225, 225, 225)
AXIS = (80, 80, 80)
RAW = (220, 70, 60)
SMOOTH = (40, 105, 220)
TEXT = (35, 35, 35)


def split_action(action: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    left = action[:, :10]
    right = action[:, 10:20]
    return {
        "left": {
            "position": left[:, :3],
            "rotation6d": left[:, 3:9],
            "width": left[:, 9:10],
        },
        "right": {
            "position": right[:, :3],
            "rotation6d": right[:, 3:9],
            "width": right[:, 9:10],
        },
    }


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.6, color: tuple[int, int, int] = TEXT, thickness: int = 1) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def panel_bounds(row: int, col: int, *, margin: int, gap: int, panel_w: int, panel_h: int, title_h: int) -> tuple[int, int, int, int]:
    x0 = margin + col * (panel_w + gap)
    y0 = margin + title_h + row * (panel_h + gap)
    return x0, y0, x0 + panel_w, y0 + panel_h


def scale_series(values: np.ndarray, y_min: float, y_max: float, y_top: int, y_bottom: int) -> np.ndarray:
    if abs(y_max - y_min) < 1e-12:
        return np.full(values.shape, (y_top + y_bottom) / 2.0)
    return y_bottom - (values - y_min) / (y_max - y_min) * (y_bottom - y_top)


def draw_line_plot(
    img: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    *,
    y_label: str = "",
) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), PANEL_BG, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), AXIS, 1)

    pad_l, pad_r, pad_t, pad_b = 52, 14, 34, 30
    px0, px1 = x0 + pad_l, x1 - pad_r
    py0, py1 = y0 + pad_t, y1 - pad_b

    combined = np.concatenate([raw, smooth])
    y_min = float(np.nanmin(combined))
    y_max = float(np.nanmax(combined))
    span = y_max - y_min
    if span < 1e-9:
        span = 1.0
    y_min -= 0.08 * span
    y_max += 0.08 * span

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(round(py1 - frac * (py1 - py0)))
        cv2.line(img, (px0, y), (px1, y), GRID, 1)
    cv2.line(img, (px0, py1), (px1, py1), AXIS, 1)
    cv2.line(img, (px0, py0), (px0, py1), AXIS, 1)

    n = len(raw)
    xs = np.linspace(px0, px1, n)
    raw_y = scale_series(raw, y_min, y_max, py0, py1)
    smooth_y = scale_series(smooth, y_min, y_max, py0, py1)
    raw_pts = np.round(np.stack([xs, raw_y], axis=1)).astype(np.int32)
    smooth_pts = np.round(np.stack([xs, smooth_y], axis=1)).astype(np.int32)
    cv2.polylines(img, [raw_pts], False, RAW, 1, cv2.LINE_AA)
    cv2.polylines(img, [smooth_pts], False, SMOOTH, 2, cv2.LINE_AA)

    draw_text(img, title, (x0 + 12, y0 + 23), scale=0.58, thickness=1)
    draw_text(img, f"{y_max:.3f}", (x0 + 5, py0 + 5), scale=0.42, color=AXIS)
    draw_text(img, f"{y_min:.3f}", (x0 + 5, py1), scale=0.42, color=AXIS)
    if y_label:
        draw_text(img, y_label, (x0 + 5, y0 + 52), scale=0.42, color=AXIS)


def draw_group(
    img: np.ndarray,
    raw_values: np.ndarray,
    smooth_values: np.ndarray,
    names: list[str],
    start_row: int,
    col: int,
    hand: str,
    group: str,
    *,
    margin: int,
    gap: int,
    panel_w: int,
    panel_h: int,
    title_h: int,
) -> None:
    for i, name in enumerate(names):
        rect = panel_bounds(start_row + i, col, margin=margin, gap=gap, panel_w=panel_w, panel_h=panel_h, title_h=title_h)
        draw_line_plot(
            img,
            raw_values[:, i],
            smooth_values[:, i],
            rect,
            f"{hand} {group} {name}",
        )


def visualize(args: argparse.Namespace) -> None:
    data = np.load(args.npz)
    raw = split_action(data["action_raw"])
    smooth = split_action(data["action_smooth"])

    rows = 10
    cols = 2
    panel_w = args.panel_width
    panel_h = args.panel_height
    gap = 18
    margin = 28
    title_h = 92
    width = margin * 2 + cols * panel_w + (cols - 1) * gap
    height = margin * 2 + title_h + rows * panel_h + (rows - 1) * gap
    img = np.full((height, width, 3), BG, dtype=np.uint8)

    draw_text(img, f"Smoothing effect: {args.npz}", (margin, 36), scale=0.8, thickness=2)
    draw_text(img, "raw", (margin, 70), scale=0.62, color=RAW, thickness=2)
    draw_text(img, "smooth", (margin + 70, 70), scale=0.62, color=SMOOTH, thickness=2)
    draw_text(
        img,
        "Each hand action = position(3) + rotation6d(6) + width(1), in episode first-camera frame",
        (margin + 180, 70),
        scale=0.58,
    )

    for col, hand in enumerate(["left", "right"]):
        draw_group(
            img,
            raw[hand]["position"],
            smooth[hand]["position"],
            ["x", "y", "z"],
            0,
            col,
            hand,
            "pos",
            margin=margin,
            gap=gap,
            panel_w=panel_w,
            panel_h=panel_h,
            title_h=title_h,
        )
        draw_group(
            img,
            raw[hand]["rotation6d"],
            smooth[hand]["rotation6d"],
            ["r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5"],
            3,
            col,
            hand,
            "rot6d",
            margin=margin,
            gap=gap,
            panel_w=panel_w,
            panel_h=panel_h,
            title_h=title_h,
        )
        draw_group(
            img,
            raw[hand]["width"],
            smooth[hand]["width"],
            ["width"],
            9,
            col,
            hand,
            "",
            margin=margin,
            gap=gap,
            panel_w=panel_w,
            panel_h=panel_h,
            title_h=title_h,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img)
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw vs smoothed EgoDex gripper actions.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-width", type=int, default=700)
    parser.add_argument("--panel-height", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
