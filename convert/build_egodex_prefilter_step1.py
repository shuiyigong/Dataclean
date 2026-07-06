from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .action_alignment import gripper_action_10d, retarget_to_gripper
    from .egodex_loader import iter_episode_hdf5, load_episode
    from .extract_keypoints import extract_bimanual_keypoints
    from .smoothing import smooth_gripper
except ImportError:
    from action_alignment import gripper_action_10d, retarget_to_gripper
    from egodex_loader import iter_episode_hdf5, load_episode
    from extract_keypoints import extract_bimanual_keypoints
    from smoothing import smooth_gripper


DEFAULT_INPUT = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_unzipped/part1/add_remove_lid")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_part1/add_remove_lid")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _episode_id(path: Path) -> str:
    return f"episode_{int(path.stem):06d}" if path.stem.isdigit() else path.stem


def _rotation_checks(rotation: np.ndarray) -> dict[str, float]:
    eye = np.eye(3, dtype=np.float32)
    orth = np.matmul(rotation, np.swapaxes(rotation, -1, -2)) - eye
    det = np.linalg.det(rotation)
    return {
        "max_orth_error": float(np.max(np.abs(orth))),
        "min_det": float(np.min(det)),
        "max_det": float(np.max(det)),
    }


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


def _to_first_camera_frame(
    joints_world: dict[str, np.ndarray],
    camera_extrinsics_world: np.ndarray,
) -> dict[str, np.ndarray]:
    """Express every per-frame joint pose in the episode's first camera frame."""

    world_to_first_camera = np.linalg.inv(camera_extrinsics_world[0]).astype(np.float32)
    return {
        name: np.einsum("ij,tjk->tik", world_to_first_camera, tf).astype(np.float32)
        for name, tf in joints_world.items()
    }


def process_episode(hdf5_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    joints_world, confidence, meta = load_episode(hdf5_path, camera_frame=False)
    joints = _to_first_camera_frame(joints_world, meta["camera_extrinsics_world"])
    keypoints = extract_bimanual_keypoints(joints)

    raw = {}
    smoothed = {}
    action_parts = []
    raw_action_parts = []
    valid_masks = []
    keypoint_valid_masks = {}
    checks = {}
    for hand in ["left", "right"]:
        hand_keypoints = np.stack([keypoints[hand][k] for k in ["thumb", "index", "middle", "wrist"]], axis=1)
        keypoint_valid_masks[hand] = _valid_keypoint_rows(hand_keypoints)
        gripper = retarget_to_gripper(keypoints[hand], hand)
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

    action_raw = np.concatenate(raw_action_parts, axis=-1).astype(np.float32)
    action = np.concatenate(action_parts, axis=-1).astype(np.float32)
    camera_pose_valid = _valid_pose_rows(meta["camera_extrinsics_world"])
    keep_mask = np.logical_and.reduce([camera_pose_valid, *keypoint_valid_masks.values(), *valid_masks])
    timestamps = np.arange(meta["num_frames"], dtype=np.float32) / float(meta["fps"])

    conf_arrays = {}
    for hand in ["left", "right"]:
        for joint in [
            f"{hand}ThumbTip",
            f"{hand}IndexFingerTip",
            f"{hand}MiddleFingerTip",
            f"{hand}Hand",
        ]:
            conf_arrays[joint] = confidence.get(joint, np.zeros(meta["num_frames"], dtype=np.float32))

    arrays = {
        "timestamps": timestamps,
        "action_raw": action_raw,
        "action_smooth": action,
        "keep_mask_pre_filter": keep_mask,
        "left_keypoints_valid": keypoint_valid_masks["left"],
        "right_keypoints_valid": keypoint_valid_masks["right"],
        "camera_pose_valid": camera_pose_valid,
        "camera_extrinsics_world": meta["camera_extrinsics_world"],
        "camera_intrinsics": meta["camera_intrinsics"],
        "left_keypoints": np.stack([keypoints["left"][k] for k in ["thumb", "index", "middle", "wrist"]], axis=1),
        "right_keypoints": np.stack([keypoints["right"][k] for k in ["thumb", "index", "middle", "wrist"]], axis=1),
        "left_position": smoothed["left"]["position"],
        "right_position": smoothed["right"]["position"],
        "left_rotation": smoothed["left"]["rotation"],
        "right_rotation": smoothed["right"]["rotation"],
        "left_width": smoothed["left"]["width"],
        "right_width": smoothed["right"]["width"],
    }
    for name, values in conf_arrays.items():
        arrays[f"confidence/{name}"] = values.astype(np.float32)

    summary = {
        "episode_id": meta["episode_id"],
        "hdf5_path": meta["hdf5_path"],
        "mp4_path": meta["mp4_path"],
        "num_frames": meta["num_frames"],
        "fps": meta["fps"],
        "task_name": meta["task_name"],
        "language_description": meta["language_description"],
        "joint_frame": "episode_first_camera",
        "source_joint_frame": meta["joint_frame"],
        "action_layout": {
            "per_hand": "position(3), rotation_6d(6), width(1)",
            "bimanual": "left(10), right(10)",
            "coordinate_frame": "episode_first_camera",
            "shape": list(action.shape),
        },
        "valid_frames_pre_filter": int(keep_mask.sum()),
        "validity": {
            "left_keypoints": int(keypoint_valid_masks["left"].sum()),
            "right_keypoints": int(keypoint_valid_masks["right"].sum()),
            "camera_pose": int(camera_pose_valid.sum()),
            "combined": int(keep_mask.sum()),
        },
        "rotation_checks": checks,
    }
    return summary, arrays


def build_dataset(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = iter_episode_hdf5(input_dir)
    if args.limit is not None:
        episode_paths = episode_paths[: args.limit]
    if not episode_paths:
        raise FileNotFoundError(f"No .hdf5 episodes found under {input_dir}")

    summaries = []
    for hdf5_path in episode_paths:
        summary, arrays = process_episode(hdf5_path, args)
        out_path = output_dir / f"{_episode_id(hdf5_path)}.npz"
        np.savez_compressed(out_path, **arrays)
        summary["npz_path"] = str(out_path)
        summaries.append(summary)
        print(f"wrote {out_path} frames={summary['num_frames']} action_shape={summary['action_layout']['shape']}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_episodes": len(summaries),
        "total_frames": int(sum(item["num_frames"] for item in summaries)),
        "coordinate_note": (
            "Raw EgoDex transforms are ARKit-origin/world-frame; output keypoints/actions "
            "are expressed in the episode's first camera frame. Per-frame camera world "
            "extrinsics are preserved as camera_extrinsics_world."
        ),
        "smoothing": {
            "position_window": args.position_window,
            "width_window": args.width_window,
            "polyorder": args.polyorder,
            "rot_sigma": args.rot_sigma,
        },
        "episodes": summaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"wrote manifest {output_dir / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EgoDex pre-filter action-alignment outputs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--position-window", type=int, default=11)
    parser.add_argument("--width-window", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--rot-sigma", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
