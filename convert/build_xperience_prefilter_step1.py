from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is only needed for optional video metadata.
    cv2 = None

try:
    from .action_alignment import gripper_action_10d, retarget_to_gripper
    from .smoothing import smooth_gripper
except ImportError:
    from action_alignment import gripper_action_10d, retarget_to_gripper
    from smoothing import smooth_gripper


DEFAULT_INPUT = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/Xperience")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/xperience_npz")
DEFAULT_VIDEO_NAME = "stereo_left.mp4"

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


def _safe_name(text: str) -> str:
    text = text.strip().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _episode_id(annotation_path: Path, input_dir: Path) -> str:
    episode_root = annotation_path.parent
    try:
        rel = episode_root.relative_to(input_dir)
    except ValueError:
        rel = Path(episode_root.name)
    name = _safe_name(str(rel))
    return name if name and name != "." else episode_root.name


def _iter_annotation_paths(input_dir: Path) -> list[Path]:
    if input_dir.is_file() and input_dir.name == "annotation.hdf5":
        return [input_dir]
    if (input_dir / "annotation.hdf5").is_file():
        return [input_dir / "annotation.hdf5"]
    return sorted(input_dir.rglob("annotation.hdf5"))


def _read_scalar(group: h5py.Group, key: str, default: Any = None) -> Any:
    if key not in group:
        return default
    value = np.asarray(group[key][...])
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot = np.empty((quat.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def _make_intrinsics(k4: np.ndarray, width: float = np.nan, height: float = np.nan) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(k4, dtype=np.float32).reshape(-1)[:4]
    return np.array([fx, fy, cx, cy, width, height, np.nan, np.nan, np.nan], dtype=np.float32)


def _video_metadata(video_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"exists": video_path.exists()}
    if cv2 is None or not video_path.exists():
        return out
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out
    out.update(
        {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or np.nan),
            "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    )
    cap.release()
    return out


def _build_head_camera_poses(f: h5py.File) -> tuple[np.ndarray, list[str]]:
    required = ["slam/quat_wxyz", "slam/trans_xyz", "slam/frame_names", "calibration/cam01/T_c0_b"]
    missing = [key for key in required if key not in f]
    if missing:
        raise KeyError(f"annotation.hdf5 is missing required arrays for head camera poses: {missing}")

    quat_wxyz = np.asarray(f["slam/quat_wxyz"][...], dtype=np.float64)
    trans_xyz = np.asarray(f["slam/trans_xyz"][...], dtype=np.float64)
    T_c0_b = np.asarray(f["calibration/cam01/T_c0_b"][...], dtype=np.float64)
    if quat_wxyz.shape[0] != trans_xyz.shape[0]:
        raise ValueError(f"SLAM quat/trans length mismatch: {quat_wxyz.shape} vs {trans_xyz.shape}")

    R_w2b = _quat_wxyz_to_rotmat(quat_wxyz)
    T_c2w = np.empty((quat_wxyz.shape[0], 4, 4), dtype=np.float32)
    for i in range(quat_wxyz.shape[0]):
        T_w2b = np.eye(4, dtype=np.float64)
        T_w2b[:3, :3] = R_w2b[i]
        T_w2b[:3, 3] = trans_xyz[i]
        T_w2c_i = T_c0_b @ T_w2b
        T_c2w_i = np.linalg.inv(T_w2c_i)
        T_c2w[i] = T_c2w_i.astype(np.float32)

    frame_names_ds = f["slam/frame_names"]
    frame_names = []
    for i in range(frame_names_ds.shape[0]):
        raw = np.asarray(frame_names_ds[i]).tobytes()
        frame_names.append(raw.decode("utf-8", errors="replace").strip("\x00"))
    return T_c2w, frame_names


def _valid_keypoint_rows(keypoints: np.ndarray) -> np.ndarray:
    flat = np.asarray(keypoints).reshape(keypoints.shape[0], -1)
    valid = np.isfinite(flat).all(axis=1)
    valid &= np.abs(flat).sum(axis=1) > 1e-9
    valid &= (np.abs(flat) < 1e8).all(axis=1)
    return valid


def _valid_pose_rows(T: np.ndarray) -> np.ndarray:
    flat = np.asarray(T).reshape(T.shape[0], -1)
    valid = np.isfinite(flat).all(axis=1)
    R = T[:, :3, :3]
    det = np.linalg.det(R)
    valid &= np.abs(det - 1.0) < 1e-2
    return valid


def _fill_invalid_points(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.asarray(points, dtype=np.float32).copy()
    safe[~np.isfinite(safe)] = 0.0
    if not bool(valid.any()):
        return safe
    valid_idx = np.flatnonzero(valid)
    first = valid_idx[0]
    last_valid = safe[first].copy()
    for i in range(safe.shape[0]):
        if valid[i]:
            last_valid = safe[i].copy()
        else:
            safe[i] = last_valid
    return safe


def _to_first_camera_frame(points_cam: np.ndarray, camera_extrinsics_world: np.ndarray) -> np.ndarray:
    world_to_first_camera = np.linalg.inv(camera_extrinsics_world[0]).astype(np.float64)
    points_h = np.concatenate(
        [points_cam.astype(np.float64), np.ones((*points_cam.shape[:2], 1), dtype=np.float64)],
        axis=-1,
    )
    points_world = np.einsum("tij,tkj->tki", camera_extrinsics_world.astype(np.float64), points_h)
    points_first = np.einsum("ij,tkj->tki", world_to_first_camera, points_world)[..., :3]
    return points_first.astype(np.float32)


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


def _timestamps(f: h5py.File, n: int, fallback_fps: float) -> np.ndarray:
    if "video/device_timestamp" in f:
        raw = np.asarray(f["video/device_timestamp"][...]).reshape(-1)[:n]
        if raw.size == n and np.isfinite(raw.astype(np.float64)).all():
            ts = raw.astype(np.float64)
            ts = ts - ts[0]
            scale = 1e-9 if np.nanmax(np.abs(ts)) > 1e6 else 1.0
            return (ts * scale).astype(np.float32)
    return (np.arange(n, dtype=np.float32) / float(fallback_fps)).astype(np.float32)


def _caption_summary(f: h5py.File) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("caption", "captions"):
        if key not in f:
            continue
        raw = f[key][...]
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        elif isinstance(raw, str):
            text = raw
        elif hasattr(raw, "tobytes"):
            text = raw.tobytes().decode("utf-8", errors="replace")
        else:
            text = str(raw)
        text = text.strip()
        out["caption_key"] = key
        out["caption_preview"] = text[:256]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return out
        config = data.get("config", {}) if isinstance(data, dict) else {}
        out["main_task"] = config.get("Main Task") or config.get("main_task")
        out["caption_total_frames"] = config.get("total_frames")
        return out
    return out


def process_episode(annotation_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    episode_root = annotation_path.parent
    video_path = episode_root / args.video_name
    video_meta = _video_metadata(video_path)

    with h5py.File(annotation_path, "r") as f:
        required = [
            "calibration/cam01/K",
            "hand_mocap/left_joints_3d",
            "hand_mocap/right_joints_3d",
        ]
        missing = [key for key in required if key not in f]
        if missing:
            raise KeyError(f"{annotation_path} is missing required arrays: {missing}")

        T_c2w, frame_names = _build_head_camera_poses(f)
        left_cam = np.asarray(f["hand_mocap/left_joints_3d"][...], dtype=np.float32).reshape(-1, 21, 3)
        right_cam = np.asarray(f["hand_mocap/right_joints_3d"][...], dtype=np.float32).reshape(-1, 21, 3)
        n = min(T_c2w.shape[0], left_cam.shape[0], right_cam.shape[0], len(frame_names))
        if args.num_frames is not None:
            n = min(n, args.num_frames)
        if n <= 0:
            raise ValueError(f"{annotation_path} has no aligned frames")

        T_c2w = T_c2w[:n]
        left_cam = left_cam[:n]
        right_cam = right_cam[:n]
        frame_names = frame_names[:n]

        video_length_sec = _read_scalar(f["video"], "length_sec") if "video" in f else None
        if video_meta.get("fps") and np.isfinite(video_meta["fps"]):
            fps = float(video_meta["fps"])
        elif video_length_sec:
            fps = float(n) / float(video_length_sec)
        else:
            fps = float(args.fps)

        width = float(video_meta.get("width") or args.image_width or np.nan)
        height = float(video_meta.get("height") or args.image_height or np.nan)
        K4 = np.asarray(f["calibration/cam01/K"][...], dtype=np.float32).reshape(-1)[:4]
        camera_intrinsics = _make_intrinsics(K4, width=width, height=height)
        timestamps = _timestamps(f, n, fps)

        video_frame_number = (
            np.asarray(f["video/frame_number"][...], dtype=np.int64).reshape(-1)[:n]
            if "video/frame_number" in f
            else np.arange(n, dtype=np.int64)
        )
        if "video/device_timestamp" in f:
            video_device_timestamp_raw = np.asarray(f["video/device_timestamp"][...]).reshape(-1)[:n]
            if np.issubdtype(video_device_timestamp_raw.dtype, np.bytes_):
                video_device_timestamp = np.char.decode(video_device_timestamp_raw.astype("S"), "utf-8")
            else:
                video_device_timestamp = video_device_timestamp_raw.astype(np.float64)
        else:
            video_device_timestamp = np.array([], dtype=np.float32)

        camera_pose_valid = _valid_pose_rows(T_c2w)
        if not bool(camera_pose_valid[0]):
            raise ValueError(f"{annotation_path} has invalid first camera pose")

        left_first_camera = _to_first_camera_frame(left_cam, T_c2w)
        right_first_camera = _to_first_camera_frame(right_cam, T_c2w)
        left_keypoints = np.stack([left_first_camera[:, i] for i in [4, 8, 12, 0]], axis=1).astype(np.float32)
        right_keypoints = np.stack([right_first_camera[:, i] for i in [4, 8, 12, 0]], axis=1).astype(np.float32)
        left_valid = _valid_keypoint_rows(left_keypoints)
        right_valid = _valid_keypoint_rows(right_keypoints)

        raw = {}
        smoothed = {}
        raw_action_parts = []
        action_parts = []
        gripper_valid_masks = []
        checks = {}
        for hand, points in [("left", left_first_camera), ("right", right_first_camera)]:
            point_valid = left_valid if hand == "left" else right_valid
            point_valid = point_valid & camera_pose_valid
            safe_points = _fill_invalid_points(points, point_valid)
            gripper = retarget_to_gripper(_extract_gripper_keypoints(safe_points), hand)
            gripper_smooth = smooth_gripper(
                gripper,
                position_window=args.position_window,
                width_window=args.width_window,
                polyorder=args.polyorder,
                rot_sigma=args.rot_sigma,
            )
            gripper["valid_mask"] &= point_valid
            gripper_smooth["valid_mask"] &= point_valid
            raw[hand] = gripper
            smoothed[hand] = gripper_smooth
            raw_action_parts.append(gripper_action_10d(gripper))
            action_parts.append(gripper_action_10d(gripper_smooth))
            gripper_valid_masks.append(gripper["valid_mask"])
            checks[f"{hand}_raw"] = _rotation_checks(gripper["rotation"])
            checks[f"{hand}_smooth"] = _rotation_checks(gripper_smooth["rotation"])

        action_raw = np.concatenate(raw_action_parts, axis=-1).astype(np.float32)
        action = np.concatenate(action_parts, axis=-1).astype(np.float32)
        keep_mask = np.logical_and.reduce([left_valid, right_valid, camera_pose_valid, *gripper_valid_masks])

        arrays: dict[str, np.ndarray] = {
            "timestamps": timestamps.astype(np.float32),
            "frame_names": np.asarray(frame_names, dtype=str),
            "video_frame_number": video_frame_number,
            "video_device_timestamp": video_device_timestamp,
            "camera_intrinsics": camera_intrinsics.astype(np.float32),
            "camera_extrinsics_world": T_c2w.astype(np.float32),
            "left_hand_21_cam": left_cam.astype(np.float32),
            "right_hand_21_cam": right_cam.astype(np.float32),
            "left_hand_21_first_camera": left_first_camera.astype(np.float32),
            "right_hand_21_first_camera": right_first_camera.astype(np.float32),
            "left_keypoints": left_keypoints,
            "right_keypoints": right_keypoints,
            "action_raw": action_raw,
            "action_smooth": action,
            "keep_mask_pre_filter": keep_mask,
            "left_keypoints_valid": left_valid,
            "right_keypoints_valid": right_valid,
            "camera_pose_valid": camera_pose_valid,
            "left_position": smoothed["left"]["position"],
            "right_position": smoothed["right"]["position"],
            "left_rotation": smoothed["left"]["rotation"],
            "right_rotation": smoothed["right"]["rotation"],
            "left_width": smoothed["left"]["width"],
            "right_width": smoothed["right"]["width"],
        }

        if args.copy_video and video_path.exists():
            copied_video = args.output_dir / f"{_episode_id(annotation_path, args.input_dir)}{video_path.suffix}"
            copied_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, copied_video)
            recorded_video_path = copied_video
        else:
            recorded_video_path = video_path

        metadata = _caption_summary(f)
        metadata_group = f["metadata"] if "metadata" in f else None
        body_height = _read_scalar(metadata_group, "body_height") if metadata_group is not None else None
        device_id = _read_scalar(metadata_group, "device_id") if metadata_group is not None else None
        device_version = _read_scalar(metadata_group, "device_version") if metadata_group is not None else None

    summary = {
        "episode_id": _episode_id(annotation_path, args.input_dir),
        "annotation_path": str(annotation_path),
        "episode_root": str(episode_root),
        "video_path": str(recorded_video_path),
        "video_name": args.video_name,
        "video_exists": bool(video_path.exists()),
        "video_metadata": video_meta,
        "num_frames": n,
        "fps": fps,
        "keypoint_order": HAND_JOINT_ORDER,
        "head_camera": "stereo_left/cam01",
        "camera_intrinsics_source": "calibration/cam01/K",
        "camera_intrinsics_layout": "camera_intrinsics = fx, fy, cx, cy, width, height, k1, k2, p1",
        "camera_pose_layout": "camera_extrinsics_world maps p_cam to p_world.",
        "hand_joint_frame": "current_frame_head_camera",
        "output_joint_frame": "episode_first_camera",
        "world_frame": (
            "Episode-local SLAM world frame. It is fixed within one episode, but unrelated across episodes. "
            "The pose loader follows HOMIE-toolkit: slam quat/trans are treated as world-to-body and "
            "calibration/cam01/T_c0_b is applied before inverting to camera-to-world."
        ),
        "body_frame": (
            "Device/body coordinate frame attached to the headset/rig. calibration/* stores camera-to-body "
            "or inter-camera transforms; SLAM changes the body pose over time relative to the fixed world frame."
        ),
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
            "camera_pose": int(camera_pose_valid.sum()),
            "combined": int(keep_mask.sum()),
        },
        "rotation_checks": checks,
        "body_height": body_height,
        "device_id": device_id,
        "device_version": device_version,
        **metadata,
    }
    return summary, arrays


def build_dataset(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotation_paths = _iter_annotation_paths(args.input_dir)
    if args.limit is not None:
        annotation_paths = annotation_paths[: args.limit]
    if not annotation_paths:
        raise FileNotFoundError(f"No annotation.hdf5 files found under {args.input_dir}")

    summaries = []
    skipped = []
    for annotation_path in annotation_paths:
        try:
            summary, arrays = process_episode(annotation_path, args)
        except Exception as exc:
            if not args.skip_invalid:
                raise
            skipped.append({"annotation_path": str(annotation_path), "reason": f"{type(exc).__name__}: {exc}"})
            print(f"skipped {annotation_path}: {type(exc).__name__}: {exc}")
            continue

        out_path = args.output_dir / f"{summary['episode_id']}.npz"
        np.savez_compressed(out_path, **arrays)
        summary["npz_path"] = str(out_path)
        summaries.append(summary)
        print(
            f"wrote {out_path} frames={summary['num_frames']} "
            f"valid={summary['valid_frames_pre_filter']} action_shape={summary['action_layout']['shape']}"
        )

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "num_episodes": len(summaries),
        "num_skipped": len(skipped),
        "total_frames": int(sum(item["num_frames"] for item in summaries)),
        "keypoint_order": HAND_JOINT_ORDER,
        "coordinate_note": (
            "Xperience hand_mocap/*_joints_3d are preserved as left_hand_21_cam/right_hand_21_cam "
            "in current-frame head-camera coordinates. left_hand_21_first_camera/right_hand_21_first_camera "
            "and action_smooth/action_raw are expressed in the episode's first camera frame. "
            "camera_extrinsics_world gives the per-frame head-camera pose in the fixed episode-local SLAM world frame."
        ),
        "recommended_extra_fields": [
            "timestamps and video_frame_number for video/sensor alignment",
            "validity masks for filtering bad hand/pose frames",
            "video_path and image size for downstream dataset packaging",
            "derived action_smooth/action_raw if the virtual gripper prefilter pipeline is reused",
        ],
        "smoothing": {
            "position_window": args.position_window,
            "width_window": args.width_window,
            "polyorder": args.polyorder,
            "rot_sigma": args.rot_sigma,
        },
        "episodes": summaries,
        "skipped_episodes": skipped,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"wrote manifest {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Xperience pre-filter NPZ files from annotation.hdf5 episodes.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-name", type=str, default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None, help="Optional per-episode frame cap for debugging.")
    parser.add_argument("--fps", type=float, default=20.0, help="Fallback FPS when video metadata and length_sec are absent.")
    parser.add_argument("--image-width", type=int, default=None, help="Fallback image width stored in camera_intrinsics.")
    parser.add_argument("--image-height", type=int, default=None, help="Fallback image height stored in camera_intrinsics.")
    parser.add_argument("--copy-video", action="store_true", help="Copy the selected head-camera video to output_dir.")
    parser.add_argument("--position-window", type=int, default=11)
    parser.add_argument("--width-window", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--rot-sigma", type=float, default=2.0)
    parser.add_argument(
        "--skip-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes missing required arrays instead of failing the whole run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
