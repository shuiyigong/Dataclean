from __future__ import annotations

import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from robot_data_processing.loader import episode_parquet_path, read_episode_canonical
from robot_data_processing.schema import DatasetSchema, HUMANOID_SCHEMA
from robot_data_processing.types import GlobalStats


def _init_histograms(state_dim: int, action_dim: int, num_bins: int) -> dict[str, np.ndarray]:
    return {
        "state_hist": np.zeros((state_dim, num_bins), dtype=np.int64),
        "action_hist": np.zeros((action_dim, num_bins), dtype=np.int64),
        "state_min": np.full(state_dim, np.inf),
        "state_max": np.full(state_dim, -np.inf),
        "action_min": np.full(action_dim, np.inf),
        "action_max": np.full(action_dim, -np.inf),
        "num_frames": np.int64(0),
    }


def _merge_histograms(acc: dict, local: dict) -> None:
    for key in ("state_hist", "action_hist", "state_min", "state_max", "action_min", "action_max"):
        if key.endswith("_hist"):
            acc[key] += local[key]
        elif key.endswith("_min"):
            acc[key] = np.minimum(acc[key], local[key])
        else:
            acc[key] = np.maximum(acc[key], local[key])
    acc["num_frames"] += local["num_frames"]


def _update_hist_block(hist: np.ndarray, values: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, num_bins: int) -> None:
    span = vmax - vmin
    span = np.where(span < 1e-12, 1.0, span)
    scaled = (values - vmin) / span
    idx = np.floor(scaled * (num_bins - 1)).astype(np.int64)
    np.clip(idx, 0, num_bins - 1, out=idx)
    for d in range(values.shape[1]):
        hist[d] += np.bincount(idx[:, d], minlength=num_bins)


def _process_stats_episode(args: tuple) -> dict | None:
    root_str, ep_idx, num_bins, action_from_state, schema = args
    path = episode_parquet_path(Path(root_str), ep_idx)
    if not path.exists():
        return None
    try:
        data = read_episode_canonical(path, schema=schema, action_from_state=action_from_state)
    except Exception:
        return None

    state = data["state"]
    action = data["action"]
    local = _init_histograms(state.shape[1], action.shape[1], num_bins)
    local["state_min"] = state.min(axis=0)
    local["state_max"] = state.max(axis=0)
    local["action_min"] = action.min(axis=0)
    local["action_max"] = action.max(axis=0)
    local["num_frames"] = np.int64(state.shape[0])
    return local


def _pass1_worker(args: tuple) -> dict | None:
    return _process_stats_episode(args)


def _pass2_worker(
    args: tuple,
    state_min: np.ndarray,
    state_max: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    num_bins: int,
) -> dict | None:
    root_str, ep_idx, _, action_from_state, schema = args
    path = episode_parquet_path(Path(root_str), ep_idx)
    if not path.exists():
        return None
    try:
        data = read_episode_canonical(path, schema=schema, action_from_state=action_from_state)
    except Exception:
        return None
    state = data["state"]
    action = data["action"]
    local = _init_histograms(state.shape[1], action.shape[1], num_bins)
    _update_hist_block(local["state_hist"], state, state_min, state_max, num_bins)
    _update_hist_block(local["action_hist"], action, action_min, action_max, num_bins)
    local["num_frames"] = np.int64(state.shape[0])
    return local


def compute_global_stats(
    dataset_root: Path,
    episode_indices: list[int],
    schema: DatasetSchema = HUMANOID_SCHEMA,
    num_workers: int = 128,
    num_bins: int = 65536,
    show_progress: bool = True,
    action_from_state: bool = False,
) -> GlobalStats:
    """Two-pass parallel histogram stats over canonical state/action."""
    state_dim = schema.pipeline_state_dim
    action_dim = schema.pipeline_action_dim
    args_list = [
        (str(dataset_root), idx, num_bins, action_from_state, schema) for idx in episode_indices
    ]

    mins = _init_histograms(state_dim, action_dim, num_bins)
    mins["num_frames"] = np.int64(0)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_pass1_worker, args_list, chunksize=32)
        if show_progress:
            iterator = tqdm(iterator, total=len(args_list), desc="stats pass1 min/max")
        for local in iterator:
            if local is None:
                continue
            _merge_histograms(mins, local)

    state_min, state_max = mins["state_min"], mins["state_max"]
    action_min, action_max = mins["action_min"], mins["action_max"]

    acc = _init_histograms(state_dim, action_dim, num_bins)
    pass2_fn = partial(
        _pass2_worker,
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
        num_bins=num_bins,
    )

    with ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(pass2_fn, args_list, chunksize=32)
        if show_progress:
            iterator = tqdm(iterator, total=len(args_list), desc="stats pass2 histogram")
        for local in iterator:
            if local is None:
                continue
            acc["state_hist"] += local["state_hist"]
            acc["action_hist"] += local["action_hist"]
            acc["num_frames"] += local["num_frames"]

    def _percentiles(hist: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, q: float) -> np.ndarray:
        out = np.zeros(hist.shape[0], dtype=np.float64)
        span = vmax - vmin
        span = np.where(span < 1e-12, 1.0, span)
        target = q / 100.0
        for d in range(hist.shape[0]):
            cdf = np.cumsum(hist[d])
            total = cdf[-1]
            if total == 0:
                out[d] = vmin[d]
                continue
            idx = int(np.searchsorted(cdf, target * total, side="left"))
            idx = min(idx, num_bins - 1)
            out[d] = vmin[d] + (idx / (num_bins - 1)) * span[d]
        return out

    return GlobalStats(
        state_q01=_percentiles(acc["state_hist"], state_min, state_max, 1.0),
        state_q99=_percentiles(acc["state_hist"], state_min, state_max, 99.0),
        action_q01=_percentiles(acc["action_hist"], action_min, action_max, 1.0),
        action_q99=_percentiles(acc["action_hist"], action_min, action_max, 99.0),
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
        num_frames=int(acc["num_frames"]),
        num_episodes=len(episode_indices),
    )


def load_or_compute_stats(
    dataset_root: Path,
    cache_path: Path,
    episode_indices: list[int],
    schema: DatasetSchema,
    recompute: bool,
    num_workers: int,
    num_bins: int,
    show_progress: bool = True,
    action_from_state: bool = False,
) -> GlobalStats:
    if cache_path.exists() and not recompute:
        loaded = GlobalStats.load(str(cache_path))
        if loaded.state_q01.shape[0] == schema.pipeline_state_dim:
            return loaded

    stats = compute_global_stats(
        dataset_root,
        episode_indices,
        schema=schema,
        num_workers=num_workers,
        num_bins=num_bins,
        show_progress=show_progress,
        action_from_state=action_from_state,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats.save(str(cache_path))
    return stats
