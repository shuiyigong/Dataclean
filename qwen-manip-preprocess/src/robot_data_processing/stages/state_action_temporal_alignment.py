from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from robot_data_processing.schema import DatasetSchema
from robot_data_processing.transforms import robomind_ur_compact_teleop


@dataclass
class StateActionAlignConfig:
    enabled: bool = True
    max_lag_frames: int = 5
    diff_epsilon: float = 1e-4
    min_active_samples: int = 10
    fixed_lag: int | None = None
    default_lag: int = 1


@dataclass
class StateActionAlignStats:
    lag_mean: int
    lag_min: int
    lag_max: int
    per_dim_lag_mean: np.ndarray
    per_dim_lag_min: np.ndarray
    per_dim_lag_max: np.ndarray
    num_episodes: int

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            lag_mean=np.array([self.lag_mean]),
            lag_min=np.array([self.lag_min]),
            lag_max=np.array([self.lag_max]),
            per_dim_lag_mean=self.per_dim_lag_mean,
            per_dim_lag_min=self.per_dim_lag_min,
            per_dim_lag_max=self.per_dim_lag_max,
            num_episodes=np.array([self.num_episodes]),
        )

    @classmethod
    def load(cls, path: str) -> StateActionAlignStats:
        data = np.load(path)
        return cls(
            lag_mean=int(data["lag_mean"][0]),
            lag_min=int(data["lag_min"][0]),
            lag_max=int(data["lag_max"][0]),
            per_dim_lag_mean=data["per_dim_lag_mean"],
            per_dim_lag_min=data["per_dim_lag_min"],
            per_dim_lag_max=data["per_dim_lag_max"],
            num_episodes=int(data["num_episodes"][0]),
        )


def compute_peak_lag_1d(
    action: np.ndarray,
    state: np.ndarray,
    max_lag_frames: int,
    diff_epsilon: float,
    min_active_samples: int,
) -> int:
    """Cross-correlate Δaction(t) with Δstate(t+lag); positive lag => state lags action."""
    da = np.diff(action)
    ds = np.diff(state)
    n = min(da.size, ds.size)
    if n < min_active_samples + max_lag_frames:
        return 0

    da = da[:n]
    ds = ds[:n]
    da_n = (da - da.mean()) / (da.std() + 1e-8)
    ds_n = (ds - ds.mean()) / (ds.std() + 1e-8)
    corr = correlate(ds_n, da_n, mode="full") / n
    lags = np.arange(-n + 1, n)
    mask = (lags >= 0) & (lags <= max_lag_frames)
    if not mask.any():
        return 0
    return int(lags[mask][np.argmax(corr[mask])])


def compute_episode_lags(
    state: np.ndarray,
    action: np.ndarray,
    cfg: StateActionAlignConfig,
    num_dims: int,
) -> np.ndarray:
    lags = np.zeros(num_dims, dtype=np.int64)
    for d in range(num_dims):
        lags[d] = compute_peak_lag_1d(
            action[:, d],
            state[:, d],
            cfg.max_lag_frames,
            cfg.diff_epsilon,
            cfg.min_active_samples,
        )
    return lags


def apply_state_action_temporal_alignment(
    state: np.ndarray,
    action: np.ndarray,
    lag: int,
    num_dims: int | None = None,
) -> np.ndarray:
    """Overwrite action[t] with state[t+lag] for matched dims (does not shift action)."""
    if lag <= 0:
        return action.copy()
    num_dims = num_dims if num_dims is not None else min(state.shape[1], action.shape[1])
    aligned = action.copy()
    limit = state.shape[0] - lag
    if limit <= 0:
        return aligned
    aligned[:limit, :num_dims] = state[lag : lag + limit, :num_dims]
    return aligned


def aggregate_lag_stats(per_episode_lags: list[np.ndarray]) -> StateActionAlignStats:
    stacked = np.stack(per_episode_lags, axis=0)
    per_dim_mean = stacked.mean(axis=0)
    return StateActionAlignStats(
        lag_mean=int(round(float(per_dim_mean.mean()))),
        lag_min=int(stacked.min()),
        lag_max=int(stacked.max()),
        per_dim_lag_mean=per_dim_mean,
        per_dim_lag_min=stacked.min(axis=0),
        per_dim_lag_max=stacked.max(axis=0),
        num_episodes=len(per_episode_lags),
    )


def resolve_alignment_lag(
    cfg: StateActionAlignConfig,
    stats: StateActionAlignStats | None,
) -> int:
    if cfg.fixed_lag is not None:
        return max(0, int(cfg.fixed_lag))
    if stats is not None:
        return max(0, stats.lag_mean)
    return max(0, cfg.default_lag)


def alignment_metadata(
    cfg: StateActionAlignConfig,
    stats: StateActionAlignStats | None,
    lag: int,
) -> dict:
    meta: dict = {
        "state_action_alignment_applied": cfg.enabled and lag > 0,
        "state_action_alignment_lag": lag,
    }
    if stats is not None:
        meta.update(
            {
                "state_action_alignment_lag_mean": stats.lag_mean,
                "state_action_alignment_lag_min": stats.lag_min,
                "state_action_alignment_lag_max": stats.lag_max,
                "state_action_alignment_per_dim_lag_mean": stats.per_dim_lag_mean.tolist(),
            }
        )
    return meta


def matched_alignment_dims(schema: DatasetSchema, state: np.ndarray, action: np.ndarray) -> int:
    return min(schema.alignment_dim, state.shape[1], action.shape[1])


def alignment_state_action_pair(
    state: np.ndarray,
    action: np.ndarray,
    schema: DatasetSchema,
) -> tuple[np.ndarray, np.ndarray]:
    if schema.embodiment == "robomind_ur":
        return robomind_ur_compact_teleop(state, action)
    num_dims = matched_alignment_dims(schema, state, action)
    return state[:, :num_dims], action[:, :num_dims]
