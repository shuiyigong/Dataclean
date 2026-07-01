from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zarr
except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific.
    zarr = None
    ZARR_IMPORT_ERROR = exc
else:
    ZARR_IMPORT_ERROR = None

try:
    from .action_alignment import gripper_action_10d, retarget_to_gripper
    from .smoothing import smooth_gripper
except ImportError:
    from action_alignment import gripper_action_10d, retarget_to_gripper
    from smoothing import smooth_gripper


DEFAULT_INPUT = Path("/mnt/project_rlinf/runze/egoverse_demo")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egoverse_demo_npz")
IMAGE_KEY = "images.front_1"
DEFAULT_ARIA_INTRINSICS = np.array(
    [736.6339, 736.6339, 960.0, 540.0, 1920.0, 1080.0, np.nan, np.nan, np.nan, np.nan],
    dtype=np.float32,
)

HAND_JOINT_ORDER = [
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb_tip",
    "index1",
    "index2",
    "index3",
    "index_tip",
    "middle1",
    "middle2",
    "middle3",
    "middle_tip",
    "ring1",
    "ring2",
    "ring3",
    "ring_tip",
    "little1",
    "little2",
    "little3",
    "little_tip",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _parse_intrinsics_arg(text: str) -> np.ndarray:
    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float32)
    if values.size == 4:
        out = np.full(10, np.nan, dtype=np.float32)
        out[:4] = values
        return out
    if values.size == 10:
        return values.astype(np.float32)
    raise argparse.ArgumentTypeError("--fallback-intrinsics expects 4 or 10 numbers")


def _episode_id(path: Path) -> str:
    stem = path.stem if path.suffix == ".tar" else path.name
    return f"episode_{int(stem):06d}" if stem.isdigit() else stem


def _require_zarr() -> None:
    if zarr is None:
        raise ImportError(
            "This script needs zarr to read EgoVerse zarr/tar episodes. "
            "Install it in the selected Python environment, e.g. `python -m pip install zarr`."
        ) from ZARR_IMPORT_ERROR


def _safe_extract_tar(tar_path: Path, target: Path) -> Path:
    target_resolved = target.resolve()
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"empty tar archive: {tar_path}")
        root_name = members[0].name.split("/")[0]
        for member in members:
            dest = (target / member.name).resolve()
            if not str(dest).startswith(str(target_resolved)):
                raise RuntimeError(f"unsafe tar member in {tar_path}: {member.name}")
        tf.extractall(target, members=members, filter="data")
    return target / root_name


def _iter_episode_roots(input_dir: Path) -> list[Path]:
    tar_paths = sorted(input_dir.glob("*.tar"))
    roots = sorted(path for path in input_dir.iterdir() if path.is_dir() and (path / "zarr.json").is_file())
    tar_stems = {path.stem for path in tar_paths}
    roots = [path for path in roots if path.name not in tar_stems]
    return tar_paths + roots


def _open_episode(path: Path, temp_root: Path | None) -> tuple[Any, Path]:
    _require_zarr()
    if path.suffix == ".tar":
        if temp_root is None:
            raise ValueError("temp_root is required for tar episodes")
        root = _safe_extract_tar(path, temp_root)
    else:
        root = path
    return zarr.open_group(str(root), mode="r"), root


def _jpeg_bytes(value: Any) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, np.ndarray):
        return value.tobytes()
    return bytes(value)


def _write_mp4(group: Any, out_path: Path, fps: int) -> str | None:
    if IMAGE_KEY not in group:
        return None

    images = group[IMAGE_KEY]
    if int(images.shape[0]) == 0:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-vcodec",
        "mjpeg",
        "-i",
        "pipe:0",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(int(images.shape[0])):
        try:
            proc.stdin.write(_jpeg_bytes(images[i]))
        except BrokenPipeError:
            break
    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
    returncode = proc.wait()
    if returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to write {out_path}: {stderr.strip()}")
    return str(out_path)


def _pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    qw, qx, qy, qz = pose[3:7]
    qnorm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if qnorm <= 1e-12:
        rot = np.eye(3, dtype=np.float64)
    else:
        qw, qx, qy, qz = qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm
        rot = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = pose[:3]
    return mat


def _valid_pose_rows(pose: np.ndarray) -> np.ndarray:
    valid = np.isfinite(pose).all(axis=1)
    valid &= np.abs(pose).sum(axis=1) > 1e-9
    valid &= (np.abs(pose) < 1e8).all(axis=1)
    qnorm = np.linalg.norm(pose[:, 3:7], axis=1)
    valid &= qnorm > 1e-9
    valid &= np.abs(qnorm - 1.0) < 0.2
    return valid


def _valid_keypoint_rows(keypoints: np.ndarray) -> np.ndarray:
    flat = keypoints.reshape(keypoints.shape[0], -1)
    valid = np.isfinite(flat).all(axis=1)
    valid &= np.abs(flat).sum(axis=1) > 1e-9
    valid &= (np.abs(flat) < 1e8).all(axis=1)
    return valid


def _to_first_camera_frame(points_world: np.ndarray, head_pose_world: np.ndarray) -> np.ndarray:
    world_to_first_camera = np.linalg.inv(_pose7_to_matrix(head_pose_world[0]))
    points_h = np.concatenate(
        [points_world.astype(np.float64), np.ones((*points_world.shape[:2], 1), dtype=np.float64)],
        axis=-1,
    )
    points_camera = np.einsum("ij,tkj->tki", world_to_first_camera, points_h)[..., :3]
    return points_camera.astype(np.float32)


def _extract_gripper_keypoints(hand_points: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "wrist": hand_points[:, 0],
        "thumb": hand_points[:, 4],
        "index": hand_points[:, 8],
        "middle": hand_points[:, 12],
    }


def _rotation_checks(rotation: np.ndarray) -> dict[str, float]:
    eye = np.eye(3, dtype=np.float32)
    orth = np.matmul(rotation, np.swapaxes(rotation, -1, -2)) - eye
    det = np.linalg.det(rotation)
    return {
        "max_orth_error": float(np.max(np.abs(orth))),
        "min_det": float(np.min(det)),
        "max_det": float(np.max(det)),
    }


def _read_attrs(group: Any) -> dict[str, Any]:
    attrs = dict(group.attrs)
    features = attrs.get("features")
    if isinstance(features, str):
        try:
            attrs["features"] = json.loads(features)
        except json.JSONDecodeError:
            pass
    return attrs


def _intrinsics_to_vector(intr: Any) -> np.ndarray:
    if intr is None:
        return np.full(10, np.nan, dtype=np.float32)
    if isinstance(intr, str):
        try:
            intr = json.loads(intr)
        except json.JSONDecodeError:
            values = np.fromstring(intr.replace(",", " "), sep=" ", dtype=np.float32)
            return _intrinsics_to_vector(values)
    if isinstance(intr, dict):
        if "camera_intrinsics" in intr:
            nested = _intrinsics_to_vector(intr["camera_intrinsics"])
            if np.isfinite(nested[:4]).all():
                return nested
        if "intrinsics" in intr:
            nested = _intrinsics_to_vector(intr["intrinsics"])
            if np.isfinite(nested[:4]).all():
                return nested
        return np.array(
            [
                intr.get("fl_x", intr.get("fx", intr.get("focal_length_x", np.nan))),
                intr.get("fl_y", intr.get("fy", intr.get("focal_length_y", np.nan))),
                intr.get("cx", intr.get("principal_point_x", np.nan)),
                intr.get("cy", intr.get("principal_point_y", np.nan)),
                intr.get("w", intr.get("width", np.nan)),
                intr.get("h", intr.get("height", np.nan)),
                intr.get("k1", np.nan),
                intr.get("k2", np.nan),
                intr.get("p1", np.nan),
                intr.get("p2", np.nan),
            ],
            dtype=np.float32,
        )
    arr = np.asarray(intr, dtype=np.float32).reshape(-1)
    if arr.size == 10:
        return arr.astype(np.float32)
    if arr.size == 9:
        mat = arr.reshape(3, 3)
        return np.array(
            [mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            dtype=np.float32,
        )
    if arr.size >= 4:
        out = np.full(10, np.nan, dtype=np.float32)
        out[: min(arr.size, 10)] = arr[:10]
        return out
    return np.full(10, np.nan, dtype=np.float32)


def _feature_image_size(attrs: dict[str, Any], group: Any) -> tuple[float, float] | None:
    features = attrs.get("features")
    if isinstance(features, dict):
        image_feature = features.get(IMAGE_KEY)
        if isinstance(image_feature, dict):
            shape = image_feature.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                return float(shape[1]), float(shape[0])
    if IMAGE_KEY in group:
        images = group[IMAGE_KEY]
        feature_shape = getattr(images, "shape", ())
        if len(feature_shape) >= 3:
            return float(feature_shape[-2]), float(feature_shape[-3])
    return None


def _scale_intrinsics_to_image(intrinsics: np.ndarray, image_size: tuple[float, float]) -> np.ndarray:
    width, height = image_size
    out = intrinsics.astype(np.float32).copy()
    source_w, source_h = out[4], out[5]
    if not np.isfinite(source_w) or source_w <= 0:
        source_w = out[2] * 2.0
    if not np.isfinite(source_h) or source_h <= 0:
        source_h = out[3] * 2.0
    if np.isfinite(source_w) and source_w > 0 and np.isfinite(source_h) and source_h > 0:
        sx = float(width) / float(source_w)
        sy = float(height) / float(source_h)
        out[0] *= sx
        out[2] *= sx
        out[1] *= sy
        out[3] *= sy
        out[4] = width
        out[5] = height
    return out


def _camera_intrinsics_from_episode(
    attrs: dict[str, Any],
    group: Any,
    fallback_intrinsics: np.ndarray,
) -> tuple[np.ndarray, str]:
    candidates = [
        ("root_attrs.intrinsics", attrs.get("intrinsics")),
        ("root_attrs.camera_intrinsics", attrs.get("camera_intrinsics")),
    ]
    for key in [IMAGE_KEY, "camera", "front_1"]:
        if key in group:
            obj_attrs = dict(getattr(group[key], "attrs", {}))
            candidates.extend(
                [
                    (f"{key}.attrs.intrinsics", obj_attrs.get("intrinsics")),
                    (f"{key}.attrs.camera_intrinsics", obj_attrs.get("camera_intrinsics")),
                ]
            )

    image_size = _feature_image_size(attrs, group)
    for source, raw in candidates:
        intrinsics = _intrinsics_to_vector(raw)
        if np.isfinite(intrinsics[:4]).all():
            if image_size is not None and (not np.isfinite(intrinsics[4:6]).all() or tuple(intrinsics[4:6]) != image_size):
                intrinsics = _scale_intrinsics_to_image(intrinsics, image_size)
                source = f"{source}, scaled_to_{int(image_size[0])}x{int(image_size[1])}"
            return intrinsics.astype(np.float32), source

    if image_size is not None:
        return _scale_intrinsics_to_image(fallback_intrinsics, image_size), (
            f"fallback_aria_intrinsics_scaled_to_{int(image_size[0])}x{int(image_size[1])}"
        )
    return fallback_intrinsics.astype(np.float32), "fallback_aria_intrinsics_unscaled"


def _read_annotations(group: Any) -> list[dict[str, Any]]:
    if "annotations" not in group:
        return []
    out = []
    for i, raw in enumerate(group["annotations"][:]):
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = {"text": text}
        if not isinstance(value, dict):
            value = {"value": value}
        value["annotation_index"] = i
        out.append(value)
    return out


def process_episode(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with tempfile.TemporaryDirectory(prefix="egoverse_demo_step1_") as tmp_s:
        group, root = _open_episode(path, Path(tmp_s))
        attrs = _read_attrs(group)
        required = [
            "left.obs_keypoints",
            "right.obs_keypoints",
            "obs_head_pose",
        ]
        missing = [key for key in required if key not in group]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")

        left_world = np.asarray(group["left.obs_keypoints"][:], dtype=np.float32).reshape(-1, 21, 3)
        right_world = np.asarray(group["right.obs_keypoints"][:], dtype=np.float32).reshape(-1, 21, 3)
        head_pose_world = np.asarray(group["obs_head_pose"][:], dtype=np.float64)
        n = min(left_world.shape[0], right_world.shape[0], head_pose_world.shape[0])
        left_world = left_world[:n]
        right_world = right_world[:n]
        head_pose_world = head_pose_world[:n]

        left = _to_first_camera_frame(left_world, head_pose_world)
        right = _to_first_camera_frame(right_world, head_pose_world)

        raw = {}
        smoothed = {}
        raw_action_parts = []
        action_parts = []
        valid_masks = []
        checks = {}
        for hand, points in [("left", left), ("right", right)]:
            gripper = retarget_to_gripper(_extract_gripper_keypoints(points), hand)
            gripper_smooth = smooth_gripper(
                gripper,
                position_window=args.position_window,
                width_window=args.width_window,
                polyorder=args.polyorder,
                rot_sigma=args.rot_sigma,
            )
            raw[hand] = gripper
            smoothed[hand] = gripper_smooth
            raw_action_parts.append(gripper_action_10d(gripper))
            action_parts.append(gripper_action_10d(gripper_smooth))
            valid_masks.append(gripper["valid_mask"])
            checks[f"{hand}_raw"] = _rotation_checks(gripper["rotation"])
            checks[f"{hand}_smooth"] = _rotation_checks(gripper_smooth["rotation"])

        left_valid = _valid_keypoint_rows(left_world)
        right_valid = _valid_keypoint_rows(right_world)
        head_valid = _valid_pose_rows(head_pose_world)
        keep_mask = np.logical_and.reduce([left_valid, right_valid, head_valid, *valid_masks])
        action_raw = np.concatenate(raw_action_parts, axis=-1).astype(np.float32)
        action = np.concatenate(action_parts, axis=-1).astype(np.float32)
        fps = int(attrs.get("fps") or args.fps)
        timestamps = np.arange(n, dtype=np.float32) / float(fps)
        video_path = _write_mp4(group, args.output_dir / f"{_episode_id(path)}.mp4", fps)

        camera_intrinsics, camera_intrinsics_source = _camera_intrinsics_from_episode(
            attrs,
            group,
            args.fallback_intrinsics,
        )

        camera_extrinsics_world = np.stack([_pose7_to_matrix(p).astype(np.float32) for p in head_pose_world], axis=0)
        arrays = {
            "timestamps": timestamps,
            "action_raw": action_raw,
            "action_smooth": action,
            "keep_mask_pre_filter": keep_mask,
            "left_keypoints_valid": left_valid,
            "right_keypoints_valid": right_valid,
            "camera_pose_valid": head_valid,
            "camera_extrinsics_world": camera_extrinsics_world,
            "camera_intrinsics": camera_intrinsics,
        }

        annotations = _read_annotations(group)
        summary = {
            "episode_id": attrs.get("episode_id") or _episode_id(path),
            "source_path": str(path),
            "extracted_root": str(root),
            "num_frames": n,
            "fps": fps,
            "embodiment": attrs.get("embodiment"),
            "task_name": attrs.get("task_name"),
            "language_description": attrs.get("task_description") or attrs.get("task_name"),
            "environment_id": attrs.get("environment_id"),
            "scene_id": attrs.get("scene_id"),
            "video_path": video_path,
            "keypoint_order": HAND_JOINT_ORDER,
            "joint_frame": "episode_first_camera",
            "source_joint_frame": "EgoVerse episode/world coordinates",
            "camera_frame_reference": "obs_head_pose[0]",
            "camera_intrinsics_source": camera_intrinsics_source,
            "camera_intrinsics_layout": "fx, fy, cx, cy, width, height, k1, k2, p1, p2",
            "action_layout": {
                "per_hand": "position(3), rotation_6d(6), width(1)",
                "bimanual": "left(10), right(10)",
                "coordinate_frame": "episode_first_camera",
                "shape": list(action.shape),
                "retarget_keypoints": "wrist(0), thumb_tip(4), index_tip(8), middle_tip(12)",
            },
            "valid_frames_pre_filter": int(keep_mask.sum()),
            "validity": {
                "left_keypoints": int(left_valid.sum()),
                "right_keypoints": int(right_valid.sum()),
                "camera_pose": int(head_valid.sum()),
            },
            "rotation_checks": checks,
            "annotations": annotations,
            "annotation_count": len(annotations),
        }
        return summary, arrays


def build_dataset(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = _iter_episode_roots(input_dir)
    if args.limit is not None:
        episode_paths = episode_paths[: args.limit]
    if not episode_paths:
        raise FileNotFoundError(f"No EgoVerse .tar or extracted zarr episodes found under {input_dir}")

    summaries = []
    for path in episode_paths:
        summary, arrays = process_episode(path, args)
        out_path = output_dir / f"{_episode_id(path)}.npz"
        np.savez_compressed(out_path, **arrays)
        summary["npz_path"] = str(out_path)
        summaries.append(summary)
        print(
            f"wrote {out_path} frames={summary['num_frames']} "
            f"valid={summary['valid_frames_pre_filter']} action_shape={summary['action_layout']['shape']}"
        )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_episodes": len(summaries),
        "total_frames": int(sum(item["num_frames"] for item in summaries)),
        "keypoint_order": HAND_JOINT_ORDER,
        "coordinate_note": (
            "Raw EgoVerse keypoints are treated as episode/world-frame 3D points. "
            "Actions are expressed in the episode's first camera frame using "
            "obs_head_pose[0]. Per-frame camera extrinsics are preserved as "
            "camera_extrinsics_world."
        ),
        "video_note": f"If {IMAGE_KEY} is present, JPEG frames are encoded to one MP4 per episode in output_dir.",
        "action_note": (
            "This is a virtual bimanual gripper retargeting from 21 hand keypoints. "
            "Each hand uses wrist, thumb_tip, index_tip, and middle_tip; the output "
            "layout matches build_pre_filter_dataset_step1.py."
        ),
        "smoothing": {
            "position_window": args.position_window,
            "width_window": args.width_window,
            "polyorder": args.polyorder,
            "rot_sigma": args.rot_sigma,
        },
        "episodes": summaries,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"wrote manifest {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EgoVerse demo pre-filter action-alignment outputs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--position-window", type=int, default=11)
    parser.add_argument("--width-window", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--rot-sigma", type=float, default=2.0)
    parser.add_argument(
        "--fallback-intrinsics",
        type=_parse_intrinsics_arg,
        default=DEFAULT_ARIA_INTRINSICS,
        help=(
            "Fallback camera intrinsics as 4 numbers (fx fy cx cy) or 10 numbers "
            "(fx fy cx cy width height k1 k2 p1 p2). Used when EgoVerse tar metadata "
            "does not contain intrinsics; scaled to the image size recorded in zarr metadata."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
