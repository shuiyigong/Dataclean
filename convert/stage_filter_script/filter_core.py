from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


EPS = 1e-8


@dataclass
class Stage1Config:
    enabled: bool = True
    smooth_window: int = 11
    median_window: int = 5
    residual_mean_multiplier: float = 5.0
    min_threshold: float = 1e-6


@dataclass
class Stage2Config:
    enabled: bool = False
    max_lag: int = 10
    min_directional_agreement: float = 0.6
    state_dims: list[int] | None = None
    action_dims: list[int] | None = None


@dataclass
class Stage3Config:
    enabled: bool = True
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    alpha: float = 0.1
    exempt_dims: list[int] = field(default_factory=list)


@dataclass
class Stage5Config:
    enabled: bool = True
    require_fixed_action_frame: bool = True
    action_frame: str = "episode_first_camera"


@dataclass
class EpisodeFilterConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)
    confidence_min: float | None = 0.15
    max_bad_frame_ratio: float = 0.2


def _odd_window(window: int, length: int) -> int:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else max(1, length - 1)
    return window


def median_filter_np(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] <= 2:
        return x.astype(np.float32)
    window = _odd_window(window, x.shape[0])
    if window <= 1:
        return x.astype(np.float32)
    half = window // 2
    padded = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        out[t] = np.median(padded[t : t + window], axis=0)
    return out.astype(np.float32)


def moving_average_np(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] <= 2:
        return x.astype(np.float32)
    window = _odd_window(window, x.shape[0])
    if window <= 1:
        return x.astype(np.float32)
    half = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    flat = padded.reshape(padded.shape[0], -1)
    out = np.empty((x.shape[0], flat.shape[1]), dtype=np.float64)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.convolve(flat[:, dim], kernel, mode="valid")
    return out.reshape(x.shape).astype(np.float32)


def pad_diff(values: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] <= order:
        return np.zeros_like(x, dtype=np.float32)
    diff = np.abs(np.diff(x, n=order, axis=0))
    left = order // 2
    right = x.shape[0] - diff.shape[0] - left
    padded = np.pad(diff, [(left, right)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return padded.astype(np.float32)


def mean_multiplier_threshold(values: np.ndarray, multiplier: float, min_threshold: float) -> np.ndarray:
    threshold = np.nanmean(np.abs(values), axis=0) * float(multiplier)
    return np.maximum(threshold, min_threshold).astype(np.float32)


def detect_sudden_changes(
    values: np.ndarray,
    config: Stage1Config,
) -> dict[str, np.ndarray]:
    trend = moving_average_np(median_filter_np(values, config.median_window), config.smooth_window)
    residual = np.abs(values - trend).astype(np.float32)
    accel = pad_diff(values, order=2)
    jerk = pad_diff(values, order=3)

    residual_threshold = mean_multiplier_threshold(
        residual,
        config.residual_mean_multiplier,
        config.min_threshold,
    )

    dim_mask = residual > residual_threshold
    frame_mask = np.any(dim_mask, axis=1)
    return {
        "frame_mask": frame_mask.astype(bool),
        "dim_mask": dim_mask.astype(bool),
        "trend": trend.astype(np.float32),
        "residual": residual,
        "accel": accel,
        "jerk": jerk,
        "residual_threshold": residual_threshold,
        "accel_threshold": np.full(values.shape[1], np.nan, dtype=np.float32),
        "jerk_threshold": np.full(values.shape[1], np.nan, dtype=np.float32),
        "residual_threshold_source": "mean_multiplier",
        "accel_threshold_source": "not_used",
        "jerk_threshold_source": "not_used",
    }


def extreme_value_bounds(
    values: np.ndarray,
    config: Stage3Config,
) -> tuple[np.ndarray, np.ndarray]:
    lower_q, upper_q = np.nanpercentile(
        values,
        [config.lower_percentile, config.upper_percentile],
        axis=0,
    )
    spread = upper_q - lower_q
    lower = lower_q - config.alpha * spread
    upper = upper_q + config.alpha * spread
    if config.exempt_dims:
        lower = lower.astype(np.float32)
        upper = upper.astype(np.float32)
        lower[config.exempt_dims] = -np.inf
        upper[config.exempt_dims] = np.inf
    return lower.astype(np.float32), upper.astype(np.float32)


def detect_extreme_values(
    values: np.ndarray,
    config: Stage3Config,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    lower, upper = bounds if bounds is not None else extreme_value_bounds(values, config)
    dim_mask = (values < lower) | (values > upper)
    frame_mask = np.any(dim_mask, axis=1)
    return {
        "frame_mask": frame_mask.astype(bool),
        "dim_mask": dim_mask.astype(bool),
        "lower": lower.astype(np.float32),
        "upper": upper.astype(np.float32),
    }


def hand_pair_confidence(confidence: np.ndarray | None) -> np.ndarray | None:
    if confidence is None:
        return None
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.ndim == 1:
        confidence = confidence[:, None]
    if confidence.shape[1] < 8:
        return np.nanmin(confidence, axis=1).astype(np.float32)

    left_wrist = confidence[:, 0]
    left_products = np.stack(
        [
            left_wrist * confidence[:, 1],
            left_wrist * confidence[:, 2],
            left_wrist * confidence[:, 3],
        ],
        axis=0,
    )
    right_wrist = confidence[:, 4]
    right_products = np.stack(
        [
            right_wrist * confidence[:, 5],
            right_wrist * confidence[:, 6],
            right_wrist * confidence[:, 7],
        ],
        axis=0,
    )
    hand_scores = np.stack(
        [
            np.nanmin(left_products, axis=0),
            np.nanmin(right_products, axis=0),
        ],
        axis=0,
    )
    return np.clip(np.nanmin(hand_scores, axis=0), 0.0, 1.0).astype(np.float32)


def confidence_keep_mask(confidence: np.ndarray | None, min_confidence: float | None) -> np.ndarray | None:
    if confidence is None or min_confidence is None:
        return None
    confidence_score = hand_pair_confidence(confidence)
    if confidence_score is None:
        return None
    return (confidence_score >= min_confidence).astype(bool)


def directional_agreement(state: np.ndarray, action: np.ndarray, max_lag: int) -> dict[str, Any]:
    if len(state) < 3 or len(action) < 3:
        return {"best_lag": 0, "score": 1.0, "scores": {0: 1.0}}

    state_diff = np.diff(state, axis=0)
    action_diff = np.diff(action, axis=0)
    scores: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s = state_diff[-lag:]
            a = action_diff[: len(s)]
        elif lag > 0:
            a = action_diff[lag:]
            s = state_diff[: len(a)]
        else:
            s = state_diff
            a = action_diff
        n = min(len(s), len(a))
        if n == 0:
            continue
        score = np.mean(np.sign(s[:n]) == np.sign(a[:n]))
        scores[lag] = float(score)

    best_lag = max(scores, key=scores.get) if scores else 0
    return {"best_lag": int(best_lag), "score": float(scores.get(best_lag, 0.0)), "scores": scores}


def check_state_action_alignment(
    state: np.ndarray | None,
    action: np.ndarray,
    config: Stage2Config,
) -> dict[str, Any]:
    if not config.enabled:
        return {"enabled": False, "passed": True, "reason": "disabled"}
    if state is None:
        return {"enabled": True, "passed": True, "reason": "missing_state"}

    state_values = state[:, config.state_dims] if config.state_dims else state
    action_values = action[:, config.action_dims] if config.action_dims else action[:, : state_values.shape[1]]
    dims = min(state_values.shape[1], action_values.shape[1])
    state_values = state_values[:, :dims]
    action_values = action_values[:, :dims]

    result = directional_agreement(state_values, action_values, config.max_lag)
    result["enabled"] = True
    result["passed"] = result["score"] >= config.min_directional_agreement
    result["min_directional_agreement"] = config.min_directional_agreement
    return result


def check_reference_frame(config: Stage5Config) -> dict[str, Any]:
    if not config.enabled:
        return {"enabled": False, "passed": True, "reason": "disabled"}
    if not config.require_fixed_action_frame:
        return {"enabled": True, "passed": True, "action_frame": config.action_frame}
    passed = config.action_frame not in {"current_camera", "per_frame_camera"}
    return {
        "enabled": True,
        "passed": passed,
        "action_frame": config.action_frame,
        "reason": "fixed_frame" if passed else "action_frame_moves_every_frame",
    }


def evaluate_episode(
    action: np.ndarray,
    *,
    config: EpisodeFilterConfig,
    state: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    stage3_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    n = len(action)
    stage1 = (
        detect_sudden_changes(action, config.stage1)
        if config.stage1.enabled
        else {"frame_mask": np.zeros(n, dtype=bool), "dim_mask": np.zeros_like(action, dtype=bool)}
    )
    stage3 = (
        detect_extreme_values(action, config.stage3, bounds=stage3_bounds)
        if config.stage3.enabled
        else {"frame_mask": np.zeros(n, dtype=bool), "dim_mask": np.zeros_like(action, dtype=bool)}
    )
    conf_keep = confidence_keep_mask(confidence, config.confidence_min)
    confidence_bad = np.zeros(n, dtype=bool) if conf_keep is None else ~conf_keep
    stage2 = check_state_action_alignment(state, action, config.stage2)
    stage5 = check_reference_frame(config.stage5)

    bad_mask = stage1["frame_mask"] | stage3["frame_mask"] | confidence_bad
    keep_mask = ~bad_mask
    bad_frame_ratio = float(np.mean(bad_mask)) if n else 1.0

    reasons: list[str] = []
    if np.mean(stage1["frame_mask"]) > config.max_bad_frame_ratio:
        reasons.append("sudden_change_high")
    if np.mean(stage3["frame_mask"]) > config.max_bad_frame_ratio:
        reasons.append("extreme_value_high")
    if np.mean(confidence_bad) > config.max_bad_frame_ratio:
        reasons.append("low_confidence")
    if bad_frame_ratio > config.max_bad_frame_ratio:
        reasons.append("bad_frame_ratio_high")
    if not stage2["passed"]:
        reasons.append("state_action_misaligned")
    if not stage5["passed"]:
        reasons.append("reference_frame_invalid")

    return {
        "keep_episode": len(reasons) == 0,
        "reasons": reasons,
        "bad_frame_ratio": bad_frame_ratio,
        "keep_mask": keep_mask.astype(bool),
        "bad_mask": bad_mask.astype(bool),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage5": stage5,
        "confidence_bad_mask": confidence_bad.astype(bool),
    }


def summarize_mask(mask: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(np.sum(mask)),
        "ratio": float(np.mean(mask)) if len(mask) else 0.0,
    }
