from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_processing.loader import filter_table_by_indices, keep_indices_from_table, valid_keep_length
from robot_data_processing.schema import stack_column


@dataclass
class EpisodeExportResult:
    episode_index: int
    original_length: int
    exported_length: int
    truncated: bool
    video_keys: list[str] = field(default_factory=list)


@dataclass
class ExportSummary:
    total_episodes: int
    total_frames: int
    truncated_episodes: int
    episodes: list[EpisodeExportResult] = field(default_factory=list)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def load_info(source_root: Path) -> dict[str, Any]:
    with (source_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def video_path(source_root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return (
        source_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def parameters_path(source_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return source_root / "parameters" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}"


def valid_prefix_length(table) -> int:
    """Backward-compatible alias for kept frame count."""
    return valid_keep_length(table)


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = prev = idx
    runs.append((start, prev + 1))
    return runs


def extract_video_by_indices(src: Path, dst: Path, indices: np.ndarray, fps: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(f"Cannot export empty video: {src}")
    if np.array_equal(indices, np.arange(indices.size)) and int(indices[0]) == 0:
        trim_video(src, dst, int(indices.size), fps)
        return

    runs = _contiguous_runs(indices)
    if len(runs) == 1:
        start, end = runs[0]
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                f"select='between(n\\,{start}\\,{end - 1})'",
                "-vsync",
                "vfr",
                "-frames:v",
                str(end - start),
                "-c",
                "copy",
                str(dst),
            ]
        )
        return

    parts = []
    concat_inputs = []
    for i, (start, end) in enumerate(runs):
        parts.append(f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{i}]")
        concat_inputs.append(f"[v{i}]")
    filter_complex = ";".join(parts) + f";{''.join(concat_inputs)}concat=n={len(runs)}:v=1:a=0[out]"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c",
            "copy",
            str(dst),
        ]
    )


def _replace_column(table, name: str, values: np.ndarray):
    idx = table.column_names.index(name)
    return table.set_column(idx, name, pa.array(values))


def filter_parquet_table(
    table,
    keep_indices: np.ndarray,
    episode_index: int,
    global_index_start: int,
    fps: float,
) -> pa.Table:
    table = filter_table_by_indices(table, keep_indices)
    if "step_validity_mask" in table.column_names:
        table = table.drop(["step_validity_mask"])

    keep_len = table.num_rows
    if keep_len == 0:
        return table

    frame_index = np.arange(keep_len, dtype=np.int64)
    timestamp = (frame_index / fps).astype(np.float32)
    episode_col = np.full(keep_len, episode_index, dtype=np.int64)
    index_col = np.arange(global_index_start, global_index_start + keep_len, dtype=np.int64)

    if "timestamps" in table.column_names:
        orig_ts = table.column("timestamps").combine_chunks().to_numpy(zero_copy_only=False)
        ts0 = int(orig_ts[0])
        if len(orig_ts) > 1:
            dt = int(orig_ts[1] - orig_ts[0])
        else:
            dt = int(round(1e9 / fps))
        timestamps = ts0 + frame_index * dt
        table = _replace_column(table, "timestamps", timestamps)

    table = _replace_column(table, "frame_index", frame_index)
    table = _replace_column(table, "timestamp", timestamp)
    table = _replace_column(table, "episode_index", episode_col)
    table = _replace_column(table, "index", index_col)
    return table


def slice_parquet_table(
    table,
    keep_len: int,
    episode_index: int,
    global_index_start: int,
    fps: float,
) -> pa.Table:
    """Legacy prefix-only slice; prefer filter_parquet_table with keep_indices."""
    return filter_parquet_table(
        table,
        np.arange(keep_len, dtype=np.int64),
        episode_index,
        global_index_start,
        fps,
    )


def probe_video_frames(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "csv=p=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    if out.isdigit() and int(out) > 0:
        return int(out)
    # Fallback: decode count (slow).
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    return int(out) if out.isdigit() else 0


def trim_video(src: Path, dst: Path, num_frames: int, fps: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if num_frames <= 0:
        raise ValueError(f"Cannot trim video to {num_frames} frames: {src}")
    # Stream copy preserves source codec and is much faster than re-encoding.
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-frames:v",
            str(num_frames),
            "-c",
            "copy",
            str(dst),
        ]
    )


def _stats_1d(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"min": [], "max": [], "mean": [], "std": [], "count": [0]}
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _stats_scalar(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"min": [], "max": [], "mean": [], "std": [], "count": [0]}
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def _video_stats_from_samples(samples: np.ndarray) -> dict[str, Any]:
    if samples.size == 0:
        return {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "count": [0],
        }
    # samples: (N, 3)
    mins, maxs, means, stds = [], [], [], []
    for c in range(3):
        col = samples[:, c]
        mins.append([[float(col.min())]])
        maxs.append([[float(col.max())]])
        means.append([[float(col.mean())]])
        stds.append([[float(col.std())]])
    return {
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "count": [int(samples.shape[0])],
    }


def sample_video_pixels(path: Path, max_samples: int = 361) -> np.ndarray:
    total = probe_video_frames(path)
    if total <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    n_samples = min(max_samples, total)
    stride = max(1, total // n_samples)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select='not(mod(n\\,{stride}))',scale=1:1",
        "-vsync",
        "vfr",
        "-frames:v",
        str(n_samples),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    if not raw:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
    return arr


def compute_episode_stats(table: pa.Table, video_paths: dict[str, Path]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        if name in video_paths:
            stats[name] = _video_stats_from_samples(sample_video_pixels(video_paths[name]))
            continue
        values = col.to_numpy(zero_copy_only=False)
        if values.dtype == object or (isinstance(values, np.ndarray) and values.dtype == object):
            arr = stack_column(values)
            stats[name] = _stats_1d(arr)
        else:
            stats[name] = _stats_scalar(np.asarray(values, dtype=np.float64))
    return stats


def _aggregate_global_stats(episode_stats: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    """Build meta/stats.json-style aggregates for state/action vectors."""
    state_key = "observation.state"
    action_key = "action"
    if not episode_stats or state_key not in episode_stats[0]:
        return {}

    def _weighted(feature: str, stat_name: str) -> list[float]:
        total = 0
        acc: list[float] = []
        for ep in episode_stats:
            count = ep[state_key]["count"][0] if feature == state_key else ep[action_key]["count"][0]
            vals = ep[feature][stat_name]
            total += count
            if not acc:
                acc = [0.0] * len(vals)
            for i, v in enumerate(vals):
                acc[i] += float(v) * count
        return [v / max(total, 1) for v in acc]

    def _global_min(feature: str) -> list[float]:
        vals = [v for ep in episode_stats for v in ep[feature]["min"]]
        if not vals:
            return []
        dim = len(episode_stats[0][feature]["min"])
        out = []
        for d in range(dim):
            out.append(min(ep[feature]["min"][d] for ep in episode_stats))
        return out

    def _global_max(feature: str) -> list[float]:
        dim = len(episode_stats[0][feature]["min"])
        return [max(ep[feature]["max"][d] for ep in episode_stats) for d in range(dim)]

    return {
        "state": {
            "min": _global_min(state_key),
            "max": _global_max(state_key),
            "mean": _weighted(state_key, "mean"),
            "std": _weighted(state_key, "std"),
        },
        "action": {
            "min": _global_min(action_key),
            "max": _global_max(action_key),
            "mean": _weighted(action_key, "mean"),
            "std": _weighted(action_key, "std"),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_lerobot_dataset(
    source_root: Path,
    filtered_root: Path,
    output_root: Path,
    episode_indices: list[int],
    *,
    num_workers: int = 8,
    recompute_video_stats: bool = True,
) -> ExportSummary:
    source_root = Path(source_root)
    filtered_root = Path(filtered_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    info = load_info(source_root)
    fps = float(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = video_keys_from_info(info)

    src_episodes = {row["episode_index"]: row for row in _load_jsonl(source_root / "meta" / "episodes.jsonl")}
    src_source_map = {
        row["merged_episode_index"]: row for row in _load_jsonl(source_root / "meta" / "episode_source_map.jsonl")
    }

    meta_out = output_root / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)

    for name in ("tasks.jsonl", "modality.json", "dreamzero_metadata.json", "dreamzero_ee_metadata.json", "dreamzero_ee_stats.json"):
        src = source_root / "meta" / name
        if src.exists():
            shutil.copy2(src, meta_out / name)

    episodes_out: list[dict[str, Any]] = []
    episodes_stats_out: list[dict[str, Any]] = []
    source_map_out: list[dict[str, Any]] = []
    export_results: list[EpisodeExportResult] = []

    global_index = 0
    total_frames = 0
    truncated_count = 0

    for episode_index in episode_indices:
        filtered_path = filtered_root / "data_filtered" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        if not filtered_path.exists():
            raise FileNotFoundError(f"Missing filtered parquet: {filtered_path}")

        table = pq.read_table(filtered_path)
        original_length = table.num_rows
        keep_indices = keep_indices_from_table(table)
        keep_len = int(keep_indices.size)
        truncated = keep_len < original_length
        if truncated:
            truncated_count += 1

        out_parquet = output_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        sliced = filter_parquet_table(table, keep_indices, episode_index, global_index, fps)
        pq.write_table(sliced, out_parquet, compression="snappy")

        out_videos: dict[str, Path] = {}
        for vk in video_keys:
            src_vid = video_path(source_root, episode_index, vk)
            dst_vid = output_root / "videos" / f"chunk-{episode_index // 1000:03d}" / vk / f"episode_{episode_index:06d}.mp4"
            if not src_vid.exists():
                raise FileNotFoundError(f"Missing source video: {src_vid}")
            extract_video_by_indices(src_vid, dst_vid, keep_indices, fps)
            out_videos[vk] = dst_vid

        src_params = parameters_path(source_root, episode_index)
        if src_params.exists():
            dst_params = output_root / "parameters" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"
            if dst_params.exists():
                shutil.rmtree(dst_params)
            shutil.copytree(src_params, dst_params)

        src_ep = src_episodes.get(episode_index, {"episode_index": episode_index, "tasks": [], "length": original_length})
        episodes_out.append(
            {
                "episode_index": episode_index,
                "tasks": src_ep.get("tasks", []),
                "length": keep_len,
            }
        )

        ep_stats = compute_episode_stats(sliced, out_videos if recompute_video_stats else {})
        episodes_stats_out.append({"episode_index": episode_index, "stats": ep_stats})

        src_map = src_source_map.get(episode_index, {})
        source_map_out.append(
            {
                "merged_episode_index": episode_index,
                "merged_global_index_start": global_index,
                "merged_global_index_end": global_index + keep_len - 1 if keep_len else global_index - 1,
                "merged_num_frames": keep_len,
                "source_repo_rel": src_map.get("source_repo_rel"),
                "source_episode_index": src_map.get("source_episode_index"),
                "source_parquet_rel": src_map.get("source_parquet_rel"),
                "merged_parquet_rel": f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet",
                "videos": {
                    vk: {
                        "src": src_map.get("videos", {}).get(vk, {}).get("src"),
                        "dst": f"videos/chunk-{episode_index // 1000:03d}/{vk}/episode_{episode_index:06d}.mp4",
                    }
                    for vk in video_keys
                },
                "parameters_copied": src_params.exists(),
                "parameters_dst_rel": f"parameters/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}",
            }
        )

        export_results.append(
            EpisodeExportResult(
                episode_index=episode_index,
                original_length=original_length,
                exported_length=keep_len,
                truncated=truncated,
                video_keys=video_keys,
            )
        )

        global_index += keep_len
        total_frames += keep_len

    _write_jsonl(meta_out / "episodes.jsonl", episodes_out)
    _write_jsonl(meta_out / "episodes_stats.jsonl", episodes_stats_out)
    _write_jsonl(meta_out / "episode_source_map.jsonl", source_map_out)

    out_info = json.loads(json.dumps(info))
    num_eps = len(episode_indices)
    out_info.update(
        {
            "total_episodes": num_eps,
            "total_frames": total_frames,
            "total_videos": num_eps * len(video_keys),
            "total_chunks": max(1, math.ceil(num_eps / chunks_size)),
            "splits": {"train": f"0:{num_eps}"},
        }
    )
    with (meta_out / "info.json").open("w", encoding="utf-8") as f:
        json.dump(out_info, f, indent=4, ensure_ascii=False)
        f.write("\n")

    global_stats = _aggregate_global_stats([row["stats"] for row in episodes_stats_out], out_info)
    if global_stats:
        with (meta_out / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(global_stats, f, indent=2, ensure_ascii=False)
            f.write("\n")

    summary = ExportSummary(
        total_episodes=num_eps,
        total_frames=total_frames,
        truncated_episodes=truncated_count,
        episodes=export_results,
    )
    with (output_root / "export_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_root": str(source_root),
                "filtered_root": str(filtered_root),
                "output_root": str(output_root),
                "total_episodes": summary.total_episodes,
                "total_frames": summary.total_frames,
                "truncated_episodes": summary.truncated_episodes,
                "episodes": [ep.__dict__ for ep in summary.episodes],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return summary


@dataclass
class AlignmentIssue:
    episode_index: int
    issue: str


def verify_lerobot_alignment(output_root: Path, episode_indices: list[int] | None = None) -> dict[str, Any]:
    output_root = Path(output_root)
    info = load_info(output_root)
    fps = float(info.get("fps", 30))
    video_keys = video_keys_from_info(info)

    episodes = _load_jsonl(output_root / "meta" / "episodes.jsonl")
    if episode_indices is None:
        episode_indices = [row["episode_index"] for row in episodes]
    ep_lengths = {row["episode_index"]: row["length"] for row in episodes}

    issues: list[AlignmentIssue] = []
    checked = 0
    expected_global = 0

    for episode_index in episode_indices:
        checked += 1
        expected_len = ep_lengths.get(episode_index)
        parquet_path = output_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        if not parquet_path.exists():
            issues.append(AlignmentIssue(episode_index, "missing parquet"))
            continue

        table = pq.read_table(parquet_path)
        n_rows = table.num_rows
        if expected_len is not None and n_rows != expected_len:
            issues.append(AlignmentIssue(episode_index, f"episodes.jsonl length={expected_len} != parquet rows={n_rows}"))

        if n_rows == 0:
            continue

        frame_index = table.column("frame_index").combine_chunks().to_numpy()
        if not np.array_equal(frame_index, np.arange(n_rows, dtype=np.int64)):
            issues.append(AlignmentIssue(episode_index, "frame_index not 0..N-1"))

        index_col = table.column("index").combine_chunks().to_numpy()
        if not np.array_equal(index_col, np.arange(expected_global, expected_global + n_rows, dtype=np.int64)):
            issues.append(AlignmentIssue(episode_index, f"global index mismatch, expected start={expected_global}"))

        episode_col = table.column("episode_index").combine_chunks().to_numpy()
        if not np.all(episode_col == episode_index):
            issues.append(AlignmentIssue(episode_index, "episode_index column mismatch"))

        ts = table.column("timestamp").combine_chunks().to_numpy()
        expected_ts = np.arange(n_rows, dtype=np.float32) / fps
        if not np.allclose(ts, expected_ts, rtol=0, atol=1e-4):
            issues.append(AlignmentIssue(episode_index, "timestamp not aligned with frame_index/fps"))

        for vk in video_keys:
            vid = output_root / "videos" / f"chunk-{episode_index // 1000:03d}" / vk / f"episode_{episode_index:06d}.mp4"
            if not vid.exists():
                issues.append(AlignmentIssue(episode_index, f"missing video {vk}"))
                continue
            n_vid = probe_video_frames(vid)
            if n_vid != n_rows:
                issues.append(AlignmentIssue(episode_index, f"{vk} frames={n_vid} != parquet rows={n_rows}"))

        expected_global += n_rows

    info_frames = int(info.get("total_frames", -1))
    info_eps = int(info.get("total_episodes", -1))
    if info_frames != expected_global:
        issues.append(AlignmentIssue(-1, f"info.total_frames={info_frames} != summed rows={expected_global}"))
    if info_eps != len(episode_indices):
        issues.append(AlignmentIssue(-1, f"info.total_episodes={info_eps} != checked={len(episode_indices)}"))

    return {
        "aligned": len(issues) == 0,
        "episodes_checked": checked,
        "total_frames": expected_global,
        "issues": [{"episode_index": i.episode_index, "issue": i.issue} for i in issues],
    }
