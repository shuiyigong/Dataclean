from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

DEFAULT_NPZ_DIR = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_part1/add_remove_lid")
DEFAULT_VIDEO_DIR = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_unzipped/part1/add_remove_lid")
DEFAULT_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_part1/add_remove_lid_lerobot_v21")
VIDEO_KEY = "observation.images.camera_top"
FPS = 30
STATE_DIM = 20
ACTION_DIM = 20
CONF_DIM = 8
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


def sorted_npz_paths(npz_dir: Path) -> list[Path]:
    return sorted(npz_dir.glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))


def episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def feature_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def features() -> dict[str, dict[str, Any]]:
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
        "observation.confidence": {
            "dtype": "float32",
            "shape": (CONF_DIM,),
            "names": [
                "left_wrist",
                "left_thumb",
                "left_index",
                "left_middle",
                "right_wrist",
                "right_thumb",
                "right_index",
                "right_middle",
            ],
        },
        "observation.camera_extrinsics_world": {
            "dtype": "float32",
            "shape": (16,),
            "names": feature_names("camera_extrinsic", 16),
        },
        "observation.camera_intrinsics": {
            "dtype": "float32",
            "shape": (9,),
            "names": feature_names("camera_intrinsic", 9),
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (3, 1080, 1920),
            "names": ["channel", "height", "width"],
        },
    }


def confidence_stack(data: np.lib.npyio.NpzFile) -> np.ndarray:
    n = len(data["action_smooth"])

    def get(name: str) -> np.ndarray:
        key = f"confidence/{name}"
        if key not in data.files:
            return np.zeros(n, dtype=np.float32)
        return data[key].astype(np.float32)

    return np.stack(
        [
            get("leftHand"),
            get("leftThumbTip"),
            get("leftIndexFingerTip"),
            get("leftMiddleFingerTip"),
            get("rightHand"),
            get("rightThumbTip"),
            get("rightIndexFingerTip"),
            get("rightMiddleFingerTip"),
        ],
        axis=1,
    ).astype(np.float32)


def task_for_episode(manifest: dict[str, Any] | None, ep_index: int) -> str:
    if manifest:
        episodes = manifest.get("episodes", [])
        if ep_index < len(episodes):
            ep = episodes[ep_index]
            return ep.get("language_description") or ep.get("task_name") or "egodex manipulation"
    return "egodex manipulation"


def write_episode_table(dataset: LeRobotDataset, episode_data: dict[str, np.ndarray], ep_index: int) -> None:
    episode_dict = {key: episode_data[key] for key in dataset.hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=dataset.hf_features, split="train")
    dataset.hf_dataset = datasets.concatenate_datasets([dataset.hf_dataset, ep_dataset])
    ep_data_path = dataset.root / dataset.meta.get_data_file_path(ep_index=ep_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def copy_episode_video(dataset: LeRobotDataset, video_dir: Path, ep_index: int) -> None:
    src = video_dir / f"{ep_index}.mp4"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = dataset.root / dataset.meta.get_video_file_path(ep_index=ep_index, vid_key=VIDEO_KEY)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert(args: argparse.Namespace) -> None:
    npz_dir = args.npz_dir.resolve()
    video_dir = args.video_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    manifest = json_load(npz_dir / "manifest.json")
    npz_paths = sorted_npz_paths(npz_dir)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No episode_*.npz files under {npz_dir}")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=features(),
        root=output_dir,
        robot_type="egodex",
        use_videos=True,
    )
    assert CODEBASE_VERSION == "v2.1"

    for npz_path in npz_paths:
        ep_index = episode_index(npz_path)
        data = np.load(npz_path)
        action = data["action_smooth"].astype(np.float32)
        if action.ndim != 2 or action.shape[1] != ACTION_DIM:
            raise ValueError(f"{npz_path} action_smooth must have shape (T, {ACTION_DIM}), got {action.shape}")
        length = int(action.shape[0])
        timestamps = data["timestamps"].astype(np.float32) if "timestamps" in data.files else np.arange(length, dtype=np.float32) / FPS
        task = task_for_episode(manifest, ep_index)

        episode_data = {
            "observation.state": action.copy(),
            "action": action,
            "observation.confidence": confidence_stack(data),
            "observation.camera_extrinsics_world": data["camera_extrinsics_world"].reshape(length, 16).astype(np.float32),
            "observation.camera_intrinsics": np.repeat(
                data["camera_intrinsics"].reshape(1, 9).astype(np.float32),
                length,
                axis=0,
            ),
            "timestamp": timestamps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep_index, dtype=np.int64),
            "index": np.arange(ds.meta.total_frames, ds.meta.total_frames + length, dtype=np.int64),
            "task_index": np.empty(length, dtype=np.int64),  # filled after task registration
        }

        if ds.meta.get_task_index(task) is None:
            ds.meta.add_task(task)
        episode_data["task_index"] = np.full(length, ds.meta.get_task_index(task), dtype=np.int64)

        write_episode_table(ds, episode_data, ep_index)
        copy_episode_video(ds, video_dir, ep_index)

        stats_features = {key: ft for key, ft in ds.features.items() if ft["dtype"] != "video"}
        ep_stats = compute_episode_stats(episode_data, stats_features)
        ds.meta.save_episode(ep_index, length, [task], ep_stats)
        print(f"converted episode {ep_index:06d}: frames={length}")

    ds.meta.update_video_info()
    from lerobot.datasets.utils import write_info

    write_info(ds.meta.info, ds.meta.root)
    print(f"Wrote LeRobot v2.1 dataset to {output_dir}")
    print(f"Episodes={ds.meta.total_episodes} frames={ds.meta.total_frames} tasks={ds.meta.total_tasks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pre-filter EgoDex npz episodes to LeRobot v2.1.")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/egodex_demo_lerobot_v21")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
