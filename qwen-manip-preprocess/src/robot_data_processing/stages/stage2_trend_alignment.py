from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from robot_data_processing.smoothing import smooth_1d


@dataclass
class Stage2Config:
    median_kernel: int = 5
    savgol_window: int = 11
    savgol_polyorder: int = 3
    max_lag_frames: int = 5
    diff_epsilon: float = 1e-4
    min_active_samples: int = 10
    da_per_dim: float = 0.65
    da_episode_mean: float = 0.65
    action_type: str = "absolute"


@dataclass
class Stage2Result:
    discard: bool
    discard_reasons: list[str]
    da_mean: float | None
    da_per_dim: list[float | None]
    lags: list[int]


def _integrate_delta(action: np.ndarray) -> np.ndarray:
    return np.cumsum(action, axis=0) + action[0]


def _compute_da_and_lag(
    state: np.ndarray,
    action: np.ndarray,
    cfg: Stage2Config,
) -> tuple[float | None, int]:
    s = smooth_1d(state, cfg.median_kernel, cfg.savgol_window, cfg.savgol_polyorder)
    a = smooth_1d(action, cfg.median_kernel, cfg.savgol_window, cfg.savgol_polyorder)

    ds = np.diff(s)
    da = np.diff(a)
    n = min(ds.size, da.size)
    if n < cfg.min_active_samples + cfg.max_lag_frames:
        return None, 0

    ds = ds[:n]
    da = da[:n]
    ds_n = (ds - ds.mean()) / (ds.std() + 1e-8)
    da_n = (da - da.mean()) / (da.std() + 1e-8)
    corr = correlate(ds_n, da_n, mode="full") / n
    lags = np.arange(-n + 1, n)
    mask = (lags >= -cfg.max_lag_frames) & (lags <= cfg.max_lag_frames)
    best_lag = int(lags[mask][np.argmax(corr[mask])])

    if best_lag >= 0:
        s1 = ds[best_lag:]
        s2 = da[: n - best_lag]
    else:
        s1 = ds[: n + best_lag]
        s2 = da[-best_lag :]

    # Only skip frames where both state and action are still; score when either moves
    # (handles recordings where state is flat but action changes).
    both_still = (np.abs(s1) <= cfg.diff_epsilon) & (np.abs(s2) <= cfg.diff_epsilon)
    active = ~both_still
    if active.sum() < cfg.min_active_samples:
        return None, best_lag
    da_score = float((np.sign(s1[active]) == np.sign(s2[active])).mean())
    return da_score, best_lag


def run_stage2(
    state_arm: np.ndarray,
    action_arm: np.ndarray,
    cfg: Stage2Config,
) -> Stage2Result:
    if cfg.action_type == "delta":
        action_arm = _integrate_delta(action_arm)

    num_dims = min(state_arm.shape[1], action_arm.shape[1])
    da_per_dim: list[float | None] = []
    lags: list[int] = []
    for d in range(num_dims):
        score, lag = _compute_da_and_lag(state_arm[:, d], action_arm[:, d], cfg)
        lags.append(lag)
        da_per_dim.append(score)

    scored = [v for v in da_per_dim if v is not None]
    da_mean = float(np.mean(scored)) if scored else None

    low_dims = [d for d, v in enumerate(da_per_dim) if v is not None and v < cfg.da_per_dim]
    discard = bool(low_dims)
    reasons: list[str] = []
    if low_dims:
        reasons.append(f"da_low_dims={low_dims}")

    return Stage2Result(
        discard=discard,
        discard_reasons=reasons,
        da_mean=da_mean,
        da_per_dim=da_per_dim,
        lags=lags,
    )
