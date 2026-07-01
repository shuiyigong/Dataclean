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

DEFAULT_NPZ_DIR = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egoverse_demo_npz")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egoverse_demo_lerobot_v21")
VIDEO_KEY = "observation.images.camera_top"
FPS = 30
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


def episode_id(path: Path) -> str:
    return path.stem


def episode_task(manifest: dict[str, Any] | None, ep_id: str) -> str:
    if manifest:
        for episode in manifest.get("episodes", []):
            if episode.get("episode_id") == ep_id or Path(episode.get("npz_path", "")).stem == ep_id:
                return episode.get("language_description") or episode.get("task_name") or "egoverse manipulation"
    return "egoverse manipulation"


def episode_video(npz_path: Path, manifest: dict[str, Any] | None) -> Path:
    ep_id = episode_id(npz_path)
    if manifest:
        for episode in manifest.get("episodes", []):
            if episode.get("episode_id") == ep_id or Path(episode.get("npz_path", "")).stem == ep_id:
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
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channel", "height", "width"],
        },
    }


def write_episode_table(dataset: LeRobotDataset, episode_data: dict[str, np.ndarray], ep_index: int) -> None:
    episode_dict = {key: episode_data[key] for key in dataset.hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=dataset.hf_features, split="train")
    dataset.hf_dataset = datasets.concatenate_datasets([dataset.hf_dataset, ep_dataset])
    ep_data_path = dataset.root / dataset.meta.get_data_file_path(ep_index=ep_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def copy_episode_video(dataset: LeRobotDataset, video_path: Path, ep_index: int) -> None:
    dst = dataset.root / dataset.meta.get_video_file_path(ep_index=ep_index, vid_key=VIDEO_KEY)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, dst)


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
        raise FileNotFoundError(f"No EgoVerse .npz files under {npz_dir}")

    first_video = episode_video(npz_paths[0], manifest)
    height, width = video_shape(first_video)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=features(height, width),
        root=output_dir,
        robot_type="egoverse",
        use_videos=True,
    )
    assert CODEBASE_VERSION == "v2.1"

    for ep_index, npz_path in enumerate(npz_paths):
        data = np.load(npz_path)
        action = data["action_smooth"].astype(np.float32)
        if action.ndim != 2 or action.shape[1] != ACTION_DIM:
            raise ValueError(f"{npz_path} action_smooth must have shape (T, {ACTION_DIM}), got {action.shape}")
        length = int(action.shape[0])
        timestamps = (
            data["timestamps"].astype(np.float32)
            if "timestamps" in data.files
            else np.arange(length, dtype=np.float32) / FPS
        )
        video_path = episode_video(npz_path, manifest)
        ep_height, ep_width = video_shape(video_path)
        if (ep_height, ep_width) != (height, width):
            raise ValueError(
                f"{video_path} has shape {(ep_height, ep_width)}, expected {(height, width)} from first video"
            )
        task = episode_task(manifest, episode_id(npz_path))

        episode_data = {
            "observation.state": action.copy(),
            "action": action.copy(),
            "timestamp": timestamps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep_index, dtype=np.int64),
            "index": np.arange(ds.meta.total_frames, ds.meta.total_frames + length, dtype=np.int64),
            "task_index": np.empty(length, dtype=np.int64),
        }

        if ds.meta.get_task_index(task) is None:
            ds.meta.add_task(task)
        episode_data["task_index"] = np.full(length, ds.meta.get_task_index(task), dtype=np.int64)

        write_episode_table(ds, episode_data, ep_index)
        copy_episode_video(ds, video_path, ep_index)

        stats_features = {key: ft for key, ft in ds.features.items() if ft["dtype"] != "video"}
        ep_stats = compute_episode_stats(episode_data, stats_features)
        ds.meta.save_episode(ep_index, length, [task], ep_stats)
        print(f"converted {episode_id(npz_path)} -> episode {ep_index:06d}: frames={length}")

    ds.meta.update_video_info()
    from lerobot.datasets.utils import write_info

    write_info(ds.meta.info, ds.meta.root)
    print(f"Wrote LeRobot v2.1 dataset to {output_dir}")
    print(f"Episodes={ds.meta.total_episodes} frames={ds.meta.total_frames} tasks={ds.meta.total_tasks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert EgoVerse step1 npz/mp4 episodes to LeRobot v2.1.")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/egoverse_demo_lerobot_v21")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
