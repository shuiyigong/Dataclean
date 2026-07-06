from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import datasets
import numpy as np

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

DEFAULT_NPZ_DIR = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/xperience_npz")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/xperience_lerobot_v21")
VIDEO_KEY = "observation.images.camera_top"
DEFAULT_FPS = 20
STATE_DIM = 20
ACTION_DIM = 20
ACTION_NAMES = [
    "left_px",
    "left_py",
    "left_pz",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_width",
    "right_px",
    "right_py",
    "right_pz",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_width",
]


def json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sorted_npz_paths(npz_dir: Path, manifest: dict[str, Any] | None) -> list[Path]:
    if manifest:
        paths = []
        for episode in manifest.get("episodes", []):
            raw = episode.get("npz_path")
            if raw:
                path = Path(raw)
                paths.append(path if path.is_absolute() else npz_dir / path)
        if paths:
            return [path for path in paths if path.exists()]
    return sorted(path for path in npz_dir.glob("*.npz") if path.name != "manifest.npz")


def episode_manifest(manifest: dict[str, Any] | None, npz_path: Path) -> dict[str, Any]:
    if not manifest:
        return {}
    ep_id = npz_path.stem
    for episode in manifest.get("episodes", []):
        raw_npz = episode.get("npz_path", "")
        if episode.get("episode_id") == ep_id or Path(raw_npz).stem == ep_id:
            return episode
    return {}


def episode_task(manifest: dict[str, Any] | None, npz_path: Path) -> str:
    episode = episode_manifest(manifest, npz_path)
    return episode.get("language_description") or episode.get("main_task") or episode.get("task_name") or "xperience manipulation"


def episode_video(npz_path: Path, manifest: dict[str, Any] | None) -> Path:
    episode = episode_manifest(manifest, npz_path)
    raw = episode.get("video_path") or episode.get("mp4_path")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else npz_path.parent / path
    return npz_path.with_suffix(".mp4")


def video_shape(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video dimensions: {video_path}")
    return height, width


def video_fps(video_path: Path, fallback: float) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if np.isfinite(fps) and fps > 0 else float(fallback)


def features(height: int, width: int) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ACTION_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ACTION_NAMES,
        },
        "observation.camera_extrinsics_world": {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"camera_extrinsic_{i}" for i in range(16)],
        },
        "observation.camera_intrinsics": {
            "dtype": "float32",
            "shape": (9,),
            "names": [f"camera_intrinsic_{i}" for i in range(9)],
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channel", "height", "width"],
        },
    }


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(changes[0::2], changes[1::2], strict=False)]


def segment_frame_numbers(data: np.lib.npyio.NpzFile, start: int, end: int) -> np.ndarray:
    if "video_frame_number" in data.files:
        frame_numbers = data["video_frame_number"].astype(np.int64).reshape(-1)
        return frame_numbers[start:end]
    return np.arange(start, end, dtype=np.int64)


def write_video_segment(
    src_video: Path,
    dst_video: Path,
    frame_numbers: np.ndarray,
    fps: float,
    expected_shape: tuple[int, int],
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src_video}")

    height, width = expected_shape
    writer = cv2.VideoWriter(str(dst_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {dst_video}")

    for frame_no in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        if not ok:
            writer.release()
            cap.release()
            dst_video.unlink(missing_ok=True)
            raise RuntimeError(f"Could not read frame {int(frame_no)} from {src_video}")
        if frame.shape[:2] != (height, width):
            writer.release()
            cap.release()
            dst_video.unlink(missing_ok=True)
            raise ValueError(f"{src_video} frame shape {frame.shape[:2]} does not match expected {(height, width)}")
        writer.write(frame)

    writer.release()
    cap.release()


def write_episode_table(dataset: LeRobotDataset, episode_data: dict[str, np.ndarray], ep_index: int) -> None:
    episode_dict = {key: episode_data[key] for key in dataset.hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=dataset.hf_features, split="train")
    dataset.hf_dataset = datasets.concatenate_datasets([dataset.hf_dataset, ep_dataset])
    ep_data_path = dataset.root / dataset.meta.get_data_file_path(ep_index=ep_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def segment_episode_data(
    data: np.lib.npyio.NpzFile,
    start: int,
    end: int,
    ep_index: int,
    task_index: int,
    global_start_index: int,
) -> dict[str, np.ndarray]:
    length = end - start
    action = data["action_smooth"][start:end].astype(np.float32)
    timestamps = (
        data["timestamps"][start:end].astype(np.float32)
        if "timestamps" in data.files
        else np.arange(start, end, dtype=np.float32) / DEFAULT_FPS
    )
    timestamps = timestamps - timestamps[0]
    intrinsics = data["camera_intrinsics"].reshape(1, 9).astype(np.float32)
    return {
        "observation.state": action.copy(),
        "action": action.copy(),
        "observation.camera_extrinsics_world": data["camera_extrinsics_world"][start:end].reshape(length, 16).astype(np.float32),
        "observation.camera_intrinsics": np.repeat(intrinsics, length, axis=0),
        "timestamp": timestamps,
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, ep_index, dtype=np.int64),
        "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
        "task_index": np.full(length, task_index, dtype=np.int64),
    }


def validate_npz(data: np.lib.npyio.NpzFile, npz_path: Path) -> int:
    action = data["action_smooth"].astype(np.float32)
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"{npz_path} action_smooth must have shape (T, {ACTION_DIM}), got {action.shape}")
    length = int(action.shape[0])
    required = ["keep_mask_pre_filter", "camera_extrinsics_world", "camera_intrinsics"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{npz_path} is missing required arrays: {missing}")
    if data["keep_mask_pre_filter"].shape[0] != length:
        raise ValueError(f"{npz_path} keep_mask_pre_filter length does not match action_smooth length")
    if data["camera_extrinsics_world"].shape[0] != length:
        raise ValueError(f"{npz_path} camera_extrinsics_world length does not match action_smooth length")
    return length


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "xperience_step2_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def convert(args: argparse.Namespace) -> None:
    npz_dir = args.npz_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    manifest = json_load(npz_dir / "manifest.json")
    npz_paths = sorted_npz_paths(npz_dir, manifest)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No Xperience .npz files under {npz_dir}")

    first_video = episode_video(npz_paths[0], manifest)
    height, width = video_shape(first_video)
    fps = int(round(args.fps or video_fps(first_video, DEFAULT_FPS)))
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features(height, width),
        root=output_dir,
        robot_type="xperience",
        use_videos=True,
    )
    assert CODEBASE_VERSION == "v2.1"

    converted_segments = 0
    skipped_episodes = []
    skipped_segments = []
    for npz_path in npz_paths:
        data = np.load(npz_path)
        length = validate_npz(data, npz_path)
        keep_mask = data["keep_mask_pre_filter"].astype(bool).reshape(-1)
        invalid_ratio = 1.0 - float(keep_mask.sum()) / float(length)
        if invalid_ratio > args.max_invalid_ratio:
            skipped_episodes.append(
                {
                    "npz_path": str(npz_path),
                    "reason": "invalid_ratio",
                    "invalid_ratio": invalid_ratio,
                    "frames": length,
                    "valid_frames": int(keep_mask.sum()),
                }
            )
            print(f"skipped episode {npz_path.stem}: invalid_ratio={invalid_ratio:.3f} frames={length}")
            continue

        src_video = episode_video(npz_path, manifest)
        ep_height, ep_width = video_shape(src_video)
        if (ep_height, ep_width) != (height, width):
            raise ValueError(f"{src_video} has shape {(ep_height, ep_width)}, expected {(height, width)} from first video")

        task = episode_task(manifest, npz_path)
        if ds.meta.get_task_index(task) is None:
            ds.meta.add_task(task)
        task_index = ds.meta.get_task_index(task)

        for seg_idx, (start, end) in enumerate(contiguous_true_segments(keep_mask)):
            seg_len = end - start
            if seg_len <= args.min_segment_frames:
                skipped_segments.append(
                    {
                        "npz_path": str(npz_path),
                        "segment_index": seg_idx,
                        "start": start,
                        "end": end,
                        "frames": seg_len,
                        "reason": "too_short",
                    }
                )
                continue

            ep_index = converted_segments
            episode_data = segment_episode_data(data, start, end, ep_index, task_index, ds.meta.total_frames)
            write_episode_table(ds, episode_data, ep_index)

            src_frame_numbers = segment_frame_numbers(data, start, end)
            dst_video = ds.root / ds.meta.get_video_file_path(ep_index=ep_index, vid_key=VIDEO_KEY)
            write_video_segment(src_video, dst_video, src_frame_numbers, fps, (height, width))

            stats_features = {key: ft for key, ft in ds.features.items() if ft["dtype"] != "video"}
            ep_stats = compute_episode_stats(episode_data, stats_features)
            ds.meta.save_episode(ep_index, seg_len, [task], ep_stats)
            converted_segments += 1
            print(
                f"converted {npz_path.stem}[{start}:{end}] -> episode {ep_index:06d}: "
                f"frames={seg_len} invalid_ratio={invalid_ratio:.3f}"
            )

    report = {
        "source_npz_dir": str(npz_dir),
        "output_dir": str(output_dir),
        "max_invalid_ratio": args.max_invalid_ratio,
        "min_segment_frames": args.min_segment_frames,
        "converted_segments": converted_segments,
        "skipped_episodes": skipped_episodes,
        "skipped_segments": skipped_segments,
    }
    write_report(output_dir, report)

    if converted_segments == 0:
        raise RuntimeError(
            "No Xperience segments were converted. See xperience_step2_report.json, or lower "
            "--min-segment-frames / raise --max-invalid-ratio."
        )

    ds.meta.update_video_info()
    from lerobot.datasets.utils import write_info

    write_info(ds.meta.info, ds.meta.root)

    print(f"Wrote LeRobot v2.1 dataset to {output_dir}")
    print(f"Episodes={ds.meta.total_episodes} frames={ds.meta.total_frames} tasks={ds.meta.total_tasks}")
    print(f"Skipped episodes={len(skipped_episodes)} skipped short segments={len(skipped_segments)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Xperience step1 npz/mp4 episodes to segmented LeRobot v2.1.")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/xperience_lerobot_v21")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-invalid-ratio", type=float, default=0.5)
    parser.add_argument("--min-segment-frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=None, help="Override output video/dataset FPS. Defaults to source video FPS.")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
