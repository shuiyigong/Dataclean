from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd

try:
    from .filter_config import (
        add_config_arg,
        add_filter_override_args,
        apply_filter_overrides,
        episode_filter_config_from_dict,
        filter_keys_from_dict,
        load_filter_config,
    )
    from .filter_core import evaluate_episode, extreme_value_bounds, summarize_mask
except ImportError:
    from filter_config import (
        add_config_arg,
        add_filter_override_args,
        apply_filter_overrides,
        episode_filter_config_from_dict,
        filter_keys_from_dict,
        load_filter_config,
    )
    from filter_core import evaluate_episode, extreme_value_bounds, summarize_mask

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import get_hf_features_from_features, write_episode_stats, write_info


DEFAULT_INPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_lerobot_v21")
DEFAULT_REPORT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/filter_reports/egodex_demo_lerobot_v21")
DEFAULT_FILTERED_OUTPUT = Path("/mnt/project_rlinf/runze/ml-egodex/convert/output/egodex_demo_lerobot_v21_filtered")
VIDEO_KEY = "observation.images.egocentric"


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_feature_shapes(info: dict[str, Any]) -> dict[str, Any]:
    for feature in info.get("features", {}).values():
        if isinstance(feature.get("shape"), list):
            feature["shape"] = tuple(feature["shape"])
    return info


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")


def reset_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def episode_chunk(ep_index: int, chunks_size: int) -> int:
    return ep_index // chunks_size


def data_file_path(info: dict[str, Any], ep_index: int) -> Path:
    return Path(
        info["data_path"].format(
            episode_chunk=episode_chunk(ep_index, info["chunks_size"]),
            episode_index=ep_index,
        )
    )


def video_file_path(info: dict[str, Any], ep_index: int, video_key: str) -> Path:
    return Path(
        info["video_path"].format(
            episode_chunk=episode_chunk(ep_index, info["chunks_size"]),
            episode_index=ep_index,
            video_key=video_key,
        )
    )


def array_column(df: pd.DataFrame, key: str) -> np.ndarray | None:
    if key not in df.columns:
        return None
    values = df[key].to_numpy()
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float32)
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values).astype(np.float32)
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist(), dtype=np.float32)
    return values.astype(np.float32)[:, None]


def dataframe_to_episode_data(df: pd.DataFrame, hf_feature_keys: list[str]) -> dict[str, np.ndarray]:
    episode_data: dict[str, np.ndarray] = {}
    for key in hf_feature_keys:
        values = df[key].to_numpy()
        first = values[0] if len(values) else None
        if isinstance(first, np.ndarray):
            episode_data[key] = np.stack(values)
        elif isinstance(first, (list, tuple)):
            episode_data[key] = np.asarray(values.tolist())
        else:
            episode_data[key] = values
    return episode_data


def collect_stage3_bounds(
    dataset_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: EpisodeFilterConfig,
    action_key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not config.stage3.enabled:
        return None
    actions = []
    for ep in episodes:
        path = dataset_root / data_file_path(info, ep["episode_index"])
        df = pd.read_parquet(path)
        action = array_column(df, action_key)
        if action is not None and len(action):
            actions.append(action)
    if not actions:
        return None
    return extreme_value_bounds(np.concatenate(actions, axis=0), config.stage3)


def compact_stage_payload(stage: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key, value in stage.items():
        if key in {"dim_mask", "trend", "residual", "accel", "jerk"}:
            continue
        compact[key] = value
    return compact


def write_episode_mask(mask_dir: Path, ep_index: int, result: dict[str, Any]) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_dir / f"episode_{ep_index:06d}.npz",
        keep_mask=result["keep_mask"],
        bad_mask=result["bad_mask"],
        stage1_sudden_change_mask=result["stage1"]["frame_mask"],
        stage3_extreme_value_mask=result["stage3"]["frame_mask"],
        confidence_bad_mask=result["confidence_bad_mask"],
        stage1_dim_mask=result["stage1"]["dim_mask"],
        stage3_dim_mask=result["stage3"]["dim_mask"],
    )


def copy_episode_to_filtered(
    *,
    src_root: Path,
    dst_root: Path,
    src_info: dict[str, Any],
    dst_info: dict[str, Any],
    src_ep: dict[str, Any],
    new_ep_index: int,
    global_index_start: int,
    task_map: dict[int, int],
    hf_features: datasets.Features,
) -> dict[str, Any]:
    src_ep_index = src_ep["episode_index"]
    df = pd.read_parquet(src_root / data_file_path(src_info, src_ep_index)).copy()
    length = len(df)
    df["episode_index"] = np.full(length, new_ep_index, dtype=np.int64)
    df["frame_index"] = np.arange(length, dtype=np.int64)
    df["index"] = np.arange(global_index_start, global_index_start + length, dtype=np.int64)
    df["task_index"] = df["task_index"].map(task_map).astype(np.int64)

    episode_data = dataframe_to_episode_data(df, list(hf_features))
    ep_dataset = datasets.Dataset.from_dict(episode_data, features=hf_features, split="train")
    dst_data_path = dst_root / data_file_path(dst_info, new_ep_index)
    dst_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(dst_data_path)

    for video_key, ft in src_info["features"].items():
        if ft.get("dtype") != "video":
            continue
        src_video = src_root / video_file_path(src_info, src_ep_index, video_key)
        dst_video = dst_root / video_file_path(dst_info, new_ep_index, video_key)
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_video, dst_video)

    stats_features = {key: ft for key, ft in dst_info["features"].items() if ft["dtype"] != "video"}
    ep_stats = compute_episode_stats(episode_data, stats_features)
    return {
        "episode": {
            "episode_index": new_ep_index,
            "tasks": src_ep["tasks"],
            "length": length,
            "source_episode_index": src_ep_index,
        },
        "stats": ep_stats,
        "length": length,
    }


def write_jsonl_file(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")


def export_filtered_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    src_info: dict[str, Any],
    episodes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    overwrite: bool,
    repo_id: str,
) -> dict[str, Any]:
    reset_path(dst_root, overwrite)

    kept_src_indices = [d["episode_index"] for d in decisions if d["keep_episode"]]
    episode_by_index = {ep["episode_index"]: ep for ep in episodes}
    kept_episodes = [episode_by_index[idx] for idx in kept_src_indices]

    task_names = []
    for ep in kept_episodes:
        for task in ep["tasks"]:
            if task not in task_names:
                task_names.append(task)
    task_to_new = {task: i for i, task in enumerate(task_names)}
    old_tasks = {item["task_index"]: item["task"] for item in load_jsonl(src_root / "meta/tasks.jsonl")}
    old_to_new_task_index = {old_idx: task_to_new[task] for old_idx, task in old_tasks.items() if task in task_to_new}

    dst_info = deepcopy(src_info)
    dst_info["repo_id"] = repo_id
    dst_info["total_episodes"] = 0
    dst_info["total_frames"] = 0
    dst_info["total_tasks"] = len(task_names)
    dst_info["total_videos"] = 0
    dst_info["total_chunks"] = 0
    dst_info["splits"] = {"train": f"0:{len(kept_episodes)}"} if kept_episodes else {"train": "0:0"}
    write_info(dst_info, dst_root)
    write_jsonl_file(
        dst_root / "meta/tasks.jsonl",
        [{"task_index": i, "task": task} for i, task in enumerate(task_names)],
    )
    write_jsonl_file(dst_root / "meta/episodes.jsonl", [])
    write_jsonl_file(dst_root / "meta/episodes_stats.jsonl", [])

    hf_features = get_hf_features_from_features(dst_info["features"])
    new_episodes = []
    all_stats = []
    global_index = 0
    for new_ep_index, src_ep in enumerate(kept_episodes):
        exported = copy_episode_to_filtered(
            src_root=src_root,
            dst_root=dst_root,
            src_info=src_info,
            dst_info=dst_info,
            src_ep=src_ep,
            new_ep_index=new_ep_index,
            global_index_start=global_index,
            task_map=old_to_new_task_index,
            hf_features=hf_features,
        )
        global_index += exported["length"]
        new_episodes.append(exported["episode"])
        all_stats.append(exported["stats"])
        write_episode_stats(new_ep_index, exported["stats"], dst_root)

    dst_info["total_episodes"] = len(new_episodes)
    dst_info["total_frames"] = int(global_index)
    dst_info["total_videos"] = len(new_episodes) * sum(1 for ft in dst_info["features"].values() if ft["dtype"] == "video")
    dst_info["total_chunks"] = 0 if not new_episodes else episode_chunk(len(new_episodes) - 1, dst_info["chunks_size"]) + 1
    dst_info["splits"] = {"train": f"0:{len(new_episodes)}"}
    write_info(dst_info, dst_root)
    write_jsonl_file(dst_root / "meta/episodes.jsonl", new_episodes)

    if all_stats:
        global_stats = aggregate_stats(all_stats)
        write_json(dst_root / "meta/stats_summary.json", {"stats": global_stats})

    return {
        "output_dir": str(dst_root),
        "kept_episodes": len(new_episodes),
        "total_frames": int(global_index),
        "repo_id": repo_id,
    }


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.resolve()
    report_dir = args.report_dir.resolve()
    reset_path(report_dir, args.overwrite_report)
    mask_dir = report_dir / "frame_masks"

    info = normalize_feature_shapes(load_json(dataset_root / "meta/info.json"))
    episodes = load_jsonl(dataset_root / "meta/episodes.jsonl")
    raw_config = apply_filter_overrides(load_filter_config(args.config), args, include_stage2=True)
    config = episode_filter_config_from_dict(raw_config)
    keys = filter_keys_from_dict(raw_config)
    if keys["confidence_key"] in {"", "none", "None", "null"}:
        keys["confidence_key"] = None
    if keys["state_key"] in {"", "none", "None", "null"}:
        keys["state_key"] = None
    stage3_bounds = collect_stage3_bounds(dataset_root, info, episodes, config, keys["action_key"])

    decisions = []
    flagged_count = 0
    for ep in episodes:
        ep_index = ep["episode_index"]
        df = pd.read_parquet(dataset_root / data_file_path(info, ep_index))
        action = array_column(df, keys["action_key"])
        if action is None:
            raise KeyError(f"{keys['action_key']} not found in episode {ep_index}")
        state = array_column(df, keys["state_key"]) if keys["state_key"] else None
        confidence = array_column(df, keys["confidence_key"]) if keys["confidence_key"] else None

        result = evaluate_episode(
            action,
            state=state,
            confidence=confidence,
            config=config,
            stage3_bounds=stage3_bounds,
        )
        write_episode_mask(mask_dir, ep_index, result)

        stage1_summary = summarize_mask(result["stage1"]["frame_mask"])
        stage3_summary = summarize_mask(result["stage3"]["frame_mask"])
        confidence_summary = summarize_mask(result["confidence_bad_mask"])
        decision = {
            "episode_index": ep_index,
            "keep_episode": result["keep_episode"],
            "bad_frame_ratio": result["bad_frame_ratio"],
            "length": int(len(action)),
            "reasons": result["reasons"],
            "stage1": stage1_summary,
            "stage2": result["stage2"],
            "stage3": stage3_summary,
            "stage5": result["stage5"],
            "confidence": confidence_summary,
        }
        decisions.append(decision)
        append_jsonl(report_dir / "episode_decisions.jsonl", decision)

        bad_indices = np.flatnonzero(result["bad_mask"])
        flagged_count += len(bad_indices)
        for frame_idx in bad_indices[: args.max_flagged_frames_per_episode]:
            append_jsonl(
                report_dir / "flagged_frames.jsonl",
                {
                    "episode_index": ep_index,
                    "frame_index": int(frame_idx),
                    "stage1": bool(result["stage1"]["frame_mask"][frame_idx]),
                    "stage3": bool(result["stage3"]["frame_mask"][frame_idx]),
                    "confidence": bool(result["confidence_bad_mask"][frame_idx]),
                },
            )

        print(
            f"episode {ep_index:06d}: keep={result['keep_episode']} "
            f"bad={result['bad_frame_ratio']:.3f} reasons={result['reasons']}"
        )

    kept = sum(1 for d in decisions if d["keep_episode"])
    summary = {
        "dataset": str(dataset_root),
        "report_dir": str(report_dir),
        "num_episodes": len(decisions),
        "kept_episodes": kept,
        "dropped_episodes": len(decisions) - kept,
        "flagged_frames": flagged_count,
        "config": config,
        "config_path": str(args.config),
        "data_keys": keys,
        "stage3_bounds": stage3_bounds,
    }
    write_json(report_dir / "filter_summary.json", summary)

    if args.filtered_output:
        export_summary = export_filtered_dataset(
            src_root=dataset_root,
            dst_root=args.filtered_output.resolve(),
            src_info=info,
            episodes=episodes,
            decisions=decisions,
            overwrite=args.overwrite_filtered,
            repo_id=args.filtered_repo_id,
        )
        summary["filtered_dataset"] = export_summary
        write_json(report_dir / "filter_summary.json", summary)

    print(f"Wrote report to {report_dir}")
    print(f"Kept {kept}/{len(decisions)} episodes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reusable Stage 1/2/3/5 filters on a LeRobot v2.1 dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--filtered-output", type=Path, default=None)
    parser.add_argument("--filtered-repo-id", default="local/egodex_demo_lerobot_v21_filtered")
    parser.add_argument("--overwrite-filtered", action="store_true")
    add_config_arg(parser)
    add_filter_override_args(parser, include_stage2=True)
    parser.add_argument("--max-flagged-frames-per-episode", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
