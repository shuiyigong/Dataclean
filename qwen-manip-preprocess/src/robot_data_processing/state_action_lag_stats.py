from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from robot_data_processing.loader import episode_parquet_path, read_episode_canonical
from robot_data_processing.schema import DatasetSchema, HUMANOID_SCHEMA
from robot_data_processing.stages.state_action_temporal_alignment import (
    StateActionAlignConfig,
    StateActionAlignStats,
    aggregate_lag_stats,
    alignment_state_action_pair,
    compute_episode_lags,
    matched_alignment_dims,
)


def _lag_worker(args: tuple) -> np.ndarray | None:
    root_str, ep_idx, cfg, schema = args
    path = episode_parquet_path(Path(root_str), ep_idx)
    if not path.exists():
        return None
    try:
        data = read_episode_canonical(path, schema=schema)
    except Exception:
        return None
    state = data["state"]
    action = data["action"]
    state_pair, action_pair = alignment_state_action_pair(state, action, schema)
    num_dims = state_pair.shape[1]
    if num_dims == 0 or state.shape[0] < cfg.min_active_samples + cfg.max_lag_frames + 1:
        return None
    return compute_episode_lags(state_pair, action_pair, cfg, num_dims)


def compute_global_action_state_lag(
    dataset_root: Path,
    episode_indices: list[int],
    schema: DatasetSchema,
    cfg: StateActionAlignConfig,
    num_workers: int = 128,
    show_progress: bool = True,
) -> StateActionAlignStats:
    args_list = [(str(dataset_root), idx, cfg, schema) for idx in episode_indices]
    per_episode: list[np.ndarray] = []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_lag_worker, args_list, chunksize=32)
        if show_progress:
            iterator = tqdm(iterator, total=len(args_list), desc="state-action lag stats")
        for lags in iterator:
            if lags is not None:
                per_episode.append(lags)
    if not per_episode:
        num_dims = schema.alignment_dim
        zeros = np.zeros(num_dims, dtype=np.float64)
        return StateActionAlignStats(
            lag_mean=cfg.default_lag,
            lag_min=cfg.default_lag,
            lag_max=cfg.default_lag,
            per_dim_lag_mean=np.full(num_dims, cfg.default_lag, dtype=np.float64),
            per_dim_lag_min=np.full(num_dims, cfg.default_lag, dtype=np.float64),
            per_dim_lag_max=np.full(num_dims, cfg.default_lag, dtype=np.float64),
            num_episodes=0,
        )
    return aggregate_lag_stats(per_episode)


def load_or_compute_action_state_lag(
    dataset_root: Path,
    cache_path: Path,
    episode_indices: list[int],
    schema: DatasetSchema,
    cfg: StateActionAlignConfig,
    recompute: bool,
    num_workers: int,
    show_progress: bool = True,
) -> StateActionAlignStats | None:
    if not cfg.enabled or schema.embodiment not in ("humanoid", "robomind_ur"):
        return None
    if cache_path.exists() and not recompute:
        loaded = StateActionAlignStats.load(str(cache_path))
        if loaded.per_dim_lag_mean.shape[0] == schema.alignment_dim:
            return loaded
    stats = compute_global_action_state_lag(
        dataset_root,
        episode_indices,
        schema,
        cfg,
        num_workers=num_workers,
        show_progress=show_progress,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats.save(str(cache_path))
    return stats
