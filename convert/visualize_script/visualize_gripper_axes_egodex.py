from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_NPZ = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_npz/episode_000020.npz")
DEFAULT_MANIFEST = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_npz/manifest.json")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/visualize/egodex/episode_000020.mp4")

AXIS_COLORS = {
    "x": (255, 40, 40),   # RGB red
    "y": (40, 220, 40),   # RGB green
    "z": (40, 120, 255),  # RGB blue
}
BAR_BG = (20, 20, 20)
BAR_BORDER = (255, 255, 255)
LEFT_BAR_COLOR = (255, 190, 40)
RIGHT_BAR_COLOR = (40, 220, 255)


def gripper_pose_to_current_camera(
    position: np.ndarray,
    rotation: np.ndarray,
    camera_extrinsics_world: np.ndarray,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a gripper pose from the first camera frame to frame_idx camera coordinates."""

    first_camera_to_world = camera_extrinsics_world[0]
    world_to_current_camera = np.linalg.inv(camera_extrinsics_world[frame_idx])
    first_camera_to_current_camera = world_to_current_camera @ first_camera_to_world
    R = first_camera_to_current_camera[:3, :3].astype(np.float32)
    t = first_camera_to_current_camera[:3, 3].astype(np.float32)
    return (R @ position + t).astype(np.float32), (R @ rotation).astype(np.float32)


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    intrinsic = intrinsic.reshape(3, 3)
    points_2d, _ = cv2.projectPoints(
        points_cam.astype(np.float32),
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        intrinsic.astype(np.float32),
        distCoeffs=np.zeros(5, dtype=np.float32),
    )
    return points_2d.reshape(-1, 2)


def camera_matrix(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics
    raise ValueError(f"Unsupported camera_intrinsics shape: {intrinsics.shape}")


def scale_camera_matrix(intrinsic: np.ndarray, intrinsics_raw: np.ndarray, width: int, height: int) -> np.ndarray:
    return intrinsic.astype(np.float32)


def gripper_tracks(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ["left_position", "right_position", "left_rotation", "right_rotation"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"EgoDex npz is missing required gripper pose fields: {missing}. Available keys: {data.files}")
    return data["left_position"], data["right_position"], data["left_rotation"], data["right_rotation"]


def draw_arrow(
    image_rgb: np.ndarray,
    start_cam: np.ndarray,
    end_cam: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if start_cam[2] <= 1e-4 or end_cam[2] <= 1e-4:
        return
    p0, p1 = project_points(np.stack([start_cam, end_cam]), intrinsic)
    h, w = image_rgb.shape[:2]
    margin = 100
    if (
        max(p0[0], p1[0]) < -margin
        or min(p0[0], p1[0]) > w + margin
        or max(p0[1], p1[1]) < -margin
        or min(p0[1], p1[1]) > h + margin
    ):
        return
    p0_i = tuple(np.round(p0).astype(int))
    p1_i = tuple(np.round(p1).astype(int))
    cv2.arrowedLine(image_rgb, p0_i, p1_i, color=color, thickness=thickness, tipLength=0.25)


def draw_frame_axes(
    image_rgb: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    intrinsic: np.ndarray,
    *,
    axis_length: float,
    thickness: int,
    label: str | None = None,
) -> None:
    if position[2] <= 1e-4:
        return
    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        end = position + axis_length * rotation[:, axis_idx]
        draw_arrow(image_rgb, position, end, intrinsic, AXIS_COLORS[axis_name], thickness)

    center_2d = project_points(position[None], intrinsic)[0]
    cv2.circle(image_rgb, tuple(np.round(center_2d).astype(int)), max(3, thickness + 1), (255, 255, 255), -1)
    if label:
        xy = tuple(np.round(center_2d + np.array([8, -8])).astype(int))
        cv2.putText(image_rgb, label, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def hand_confidence(data: np.lib.npyio.NpzFile, hand: str) -> np.ndarray:
    n = len(data[f"{hand}_position"])

    def confidence_array(joint: str) -> np.ndarray:
        key = f"confidence/{joint}"
        if key not in data.files:
            return np.zeros(n, dtype=np.float32)
        return data[key]

    wrist = confidence_array(f"{hand}Hand")
    products = np.stack(
        [
            wrist * confidence_array(f"{hand}ThumbTip"),
            wrist * confidence_array(f"{hand}IndexFingerTip"),
            wrist * confidence_array(f"{hand}MiddleFingerTip"),
        ],
        axis=0,
    )
    return np.clip(products.min(axis=0), 0.0, 1.0).astype(np.float32)


def draw_confidence_bar(
    image_rgb: np.ndarray,
    value: float,
    *,
    side: str,
    color: tuple[int, int, int],
    label: str,
) -> None:
    h, w = image_rgb.shape[:2]
    bar_height = int(h * 0.38)
    bar_width = max(18, int(w * 0.012))
    top = int(h * 0.12)
    bottom = top + bar_height
    x0 = int(w * 0.035) if side == "left" else int(w * 0.965) - bar_width
    x1 = x0 + bar_width

    value = float(np.clip(value, 0.0, 1.0))
    fill_top = bottom - int(round(value * bar_height))

    overlay = image_rgb.copy()
    cv2.rectangle(overlay, (x0, top), (x1, bottom), BAR_BG, -1)
    cv2.rectangle(overlay, (x0, fill_top), (x1, bottom), color, -1)
    cv2.rectangle(overlay, (x0, top), (x1, bottom), BAR_BORDER, 2)
    cv2.addWeighted(overlay, 0.65, image_rgb, 0.35, 0, dst=image_rgb)

    text_x = x0 - 8 if side == "left" else x0 - 20
    cv2.putText(
        image_rgb,
        label,
        (text_x, max(24, top - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BAR_BORDER,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_rgb,
        f"{value:.2f}",
        (x0 - 14, bottom + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        BAR_BORDER,
        2,
        cv2.LINE_AA,
    )


def draw_axis_legend(image_rgb: np.ndarray, *, labels: bool = True) -> None:
    h, w = image_rgb.shape[:2]
    origin = np.array([w - 120, h - 56], dtype=np.int32)
    axes = [
        ("x", np.array([42, 0], dtype=np.int32)),
        ("y", np.array([0, -42], dtype=np.int32)),
        ("z", np.array([30, 30], dtype=np.int32)),
    ]
    overlay = image_rgb.copy()
    cv2.rectangle(overlay, tuple(origin + [-20, -58]), tuple(origin + [78, 46]), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, image_rgb, 0.55, 0, dst=image_rgb)
    cv2.circle(image_rgb, tuple(origin), 4, (255, 255, 255), -1)
    for name, delta in axes:
        end = origin + delta
        color = AXIS_COLORS[name]
        cv2.arrowedLine(image_rgb, tuple(origin), tuple(end), color=color, thickness=3, tipLength=0.25)
        if labels:
            cv2.putText(
                image_rgb,
                name,
                tuple(end + np.array([4, 4], dtype=np.int32)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_status_text(image_rgb: np.ndarray, frame_idx: int, frame_name: str) -> None:
    text = f"frame {frame_idx:04d} | axes: x red, y green, z blue | gripper frame: {frame_name}"
    cv2.putText(image_rgb, text, (16, image_rgb.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)


def infer_mp4_path(npz_path: Path, manifest_path: Path | None) -> Path:
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for ep in manifest.get("episodes", []):
            ep_npz_path = Path(ep.get("npz_path", ""))
            if ep_npz_path == npz_path or ep_npz_path.name == npz_path.name:
                mp4_path = ep.get("mp4_path")
                if mp4_path:
                    return Path(mp4_path)
                return npz_path.with_suffix(".mp4")
    sibling = npz_path.with_suffix(".mp4")
    if sibling.exists():
        return sibling
    stem = npz_path.stem.removeprefix("episode_")
    if stem.isdigit():
        return Path("/mnt/project_rlinf/runze/egodex_demo") / f"{int(stem)}.mp4"
    raise ValueError(f"Could not infer mp4 path for {npz_path}; pass --mp4")


def visualize(args: argparse.Namespace) -> None:
    npz_path = args.npz.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else None
    mp4_path = args.mp4.resolve() if args.mp4 else infer_mp4_path(npz_path, manifest_path)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    intrinsics_raw = data["camera_intrinsics"]
    intrinsics = camera_matrix(intrinsics_raw)
    left_position, right_position, left_rotation, right_rotation = gripper_tracks(data)
    left_confidence = hand_confidence(data, "left")
    right_confidence = hand_confidence(data, "right")
    n = min(len(left_position), args.max_frames if args.max_frames else len(left_position))
    if args.gripper_frame == "first-camera":
        camera_extrinsics_world = data["camera_extrinsics_world"]
        if len(camera_extrinsics_world) < n:
            raise ValueError(
                f"camera_extrinsics_world has {len(camera_extrinsics_world)} frames, "
                f"but visualization needs {n} frames"
            )

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intrinsics = scale_camera_matrix(intrinsics, intrinsics_raw, width, height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while frame_idx < n:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if args.gripper_frame == "first-camera":
            left_pos, left_rot = gripper_pose_to_current_camera(
                left_position[frame_idx],
                left_rotation[frame_idx],
                camera_extrinsics_world,
                frame_idx,
            )
            right_pos, right_rot = gripper_pose_to_current_camera(
                right_position[frame_idx],
                right_rotation[frame_idx],
                camera_extrinsics_world,
                frame_idx,
            )
        else:
            left_pos, left_rot = left_position[frame_idx], left_rotation[frame_idx]
            right_pos, right_rot = right_position[frame_idx], right_rotation[frame_idx]

        draw_frame_axes(
            frame_rgb,
            left_pos,
            left_rot,
            intrinsics,
            axis_length=args.axis_length,
            thickness=args.thickness,
            label="L" if args.labels else None,
        )
        draw_frame_axes(
            frame_rgb,
            right_pos,
            right_rot,
            intrinsics,
            axis_length=args.axis_length,
            thickness=args.thickness,
            label="R" if args.labels else None,
        )
        if not args.no_confidence_bars:
            draw_confidence_bar(
                frame_rgb,
                left_confidence[frame_idx],
                side="left",
                color=LEFT_BAR_COLOR,
                label="L conf",
            )
            draw_confidence_bar(
                frame_rgb,
                right_confidence[frame_idx],
                side="right",
                color=RIGHT_BAR_COLOR,
                label="R conf",
            )
        if not args.no_legend:
            draw_axis_legend(frame_rgb)
            draw_status_text(frame_rgb, frame_idx, args.gripper_frame)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {frame_idx} frames to {output_path}")
    print(f"Source npz: {npz_path}")
    print(f"Source mp4: {mp4_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw converted left/right gripper coordinate axes on EgoDex video.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Converted episode .npz.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest used to infer the source mp4.")
    parser.add_argument("--mp4", type=Path, default=None, help="Source video. If omitted, inferred from manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--axis-length", type=float, default=0.06, help="Axis arrow length in meters.")
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--labels", action="store_true", help="Draw L/R labels at axis origins.")
    parser.add_argument("--no-confidence-bars", action="store_true", help="Disable dynamic hand confidence bars.")
    parser.add_argument("--no-legend", action="store_true", help="Disable axis legend and frame text.")
    parser.add_argument(
        "--gripper-frame",
        choices=["first-camera", "current-camera"],
        default="first-camera",
        help=(
            "Coordinate frame of left/right gripper poses in the npz. New step1 outputs use first-camera; "
            "old outputs used current-camera."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
