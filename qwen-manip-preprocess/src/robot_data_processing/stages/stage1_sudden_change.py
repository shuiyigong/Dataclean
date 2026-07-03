from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_data_processing.schema import DatasetSchema, HUMANOID_SCHEMA, max_contiguous_run
from robot_data_processing.mask import apply_per_dim_exclude
from robot_data_processing.smoothing import detect_sudden_changes_2d, detect_sudden_changes_2d_with_thresholds
from robot_data_processing.stage1_stats import Stage1GlobalStats


@dataclass
class Stage1Config:
    median_kernel: int = 5
    savgol_window: int = 11
    savgol_polyorder: int = 3
    k_residual: float = 8.0
    k_accel: float = 8.0
    k_jerk: float = 8.0
    percentile_floor: float = 99.9
    joint_abs_max: float = 3.5
    ee_position_max: float = 2.0
    rpy_abs_max: float = 3.14159265
    gripper_max: float = 1.0
    frame_max_cluster: int = 4
    frame_abnormal_min_cluster: int = 1
    episode_min_cluster: int = 20
    min_cluster_frame_jump: float = 0.08
    on_hard_limit_violation: bool = True


@dataclass
class Stage1Result:
    discard: bool
    discard_reasons: list[str]
    abnormal_frames: np.ndarray
    flagged_count: int
    hard_limit_hit: bool


def _hard_limit_abnormal_frames(
    state: np.ndarray,
    action: np.ndarray,
    cfg: Stage1Config,
    schema: DatasetSchema,
) -> tuple[np.ndarray, bool, list[str]]:
    num_frames = state.shape[0]
    abnormal = np.zeros(num_frames, dtype=bool)
    reasons: list[str] = []

    if schema.layout == "joint_gripper":
        joints = list(schema.joint_indices)
        action_joints = list(schema.action_joint_indices)
        if joints:
            abnormal |= np.any(np.abs(state[:, joints]) > cfg.joint_abs_max, axis=1)
            if np.any(np.abs(state[:, joints]) > cfg.joint_abs_max):
                reasons.append("hard_limit_state_joint")
        if action_joints:
            abnormal |= np.any(np.abs(action[:, action_joints]) > cfg.joint_abs_max, axis=1)
            if np.any(np.abs(action[:, action_joints]) > cfg.joint_abs_max):
                reasons.append("hard_limit_action_joint")
        if state.shape[1] > 14 and schema.embodiment == "humanoid":
            ee_xyz_idx = (14, 15, 16, 21, 22, 23)
            abnormal |= np.any(np.abs(state[:, ee_xyz_idx]) > cfg.ee_position_max, axis=1)
            if np.any(np.abs(state[:, ee_xyz_idx]) > cfg.ee_position_max):
                reasons.append("hard_limit_state_ee_xyz")
            quat_idx = list(range(17, 21)) + list(range(24, 28))
            abnormal |= np.any(np.abs(state[:, quat_idx]) > 1.01, axis=1)
            if np.any(np.abs(state[:, quat_idx]) > 1.01):
                reasons.append("hard_limit_state_ee_quat")
    else:
        xyz = list(schema.xyz_indices)
        if xyz:
            abnormal |= np.any(np.abs(state[:, xyz]) > cfg.ee_position_max, axis=1)
            if np.any(np.abs(state[:, xyz]) > cfg.ee_position_max):
                reasons.append("hard_limit_state_xyz")
        rpy = list(schema.rpy_indices)
        if rpy:
            abnormal |= np.any(np.abs(state[:, rpy]) > cfg.rpy_abs_max, axis=1)
            if np.any(np.abs(state[:, rpy]) > cfg.rpy_abs_max):
                reasons.append("hard_limit_state_rpy")

    for gi in schema.gripper_indices:
        abnormal |= np.abs(state[:, gi]) > cfg.gripper_max
    for gi in schema.action_gripper_indices:
        abnormal |= np.abs(action[:, gi]) > cfg.gripper_max
    if any(np.any(np.abs(state[:, gi]) > cfg.gripper_max) for gi in schema.gripper_indices):
        reasons.append("hard_limit_state_gripper")
    if any(np.any(np.abs(action[:, gi]) > cfg.gripper_max) for gi in schema.action_gripper_indices):
        reasons.append("hard_limit_action_gripper")

    return abnormal, bool(reasons), reasons


def _cluster_filter(flags: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or not flags.any():
        return flags
    out = np.zeros_like(flags)
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            out[s:e] = True
    return out


def run_stage1(
    state: np.ndarray,
    action: np.ndarray,
    cfg: Stage1Config,
    schema: DatasetSchema = HUMANOID_SCHEMA,
    startup_exclude_per_joint: np.ndarray | None = None,
    global_stats: Stage1GlobalStats | None = None,
) -> Stage1Result:
    num_frames = state.shape[0]
    reasons: list[str] = []

    abnormal, hard_hit, hard_reasons = _hard_limit_abnormal_frames(state, action, cfg, schema)
    if hard_hit:
        reasons.extend(hard_reasons)

    smooth_kwargs = dict(
        median_kernel=cfg.median_kernel,
        savgol_window=cfg.savgol_window,
        savgol_polyorder=cfg.savgol_polyorder,
    )

    discard = hard_hit and cfg.on_hard_limit_violation

    if global_stats is not None:
        state_flags = detect_sudden_changes_2d_with_thresholds(
            state,
            global_stats.thr_residual,
            global_stats.thr_accel,
            global_stats.thr_jerk,
            channel_offset=schema.stage1_state_channel_offset,
            **smooth_kwargs,
        )
        action_flags = detect_sudden_changes_2d_with_thresholds(
            action,
            global_stats.thr_residual,
            global_stats.thr_accel,
            global_stats.thr_jerk,
            channel_offset=schema.stage1_action_channel_offset,
            **smooth_kwargs,
        )
    else:
        per_ep_kwargs = dict(
            k_residual=cfg.k_residual,
            k_accel=cfg.k_accel,
            k_jerk=cfg.k_jerk,
            percentile_floor=cfg.percentile_floor,
            **smooth_kwargs,
        )
        state_flags = detect_sudden_changes_2d(state, **per_ep_kwargs)
        action_flags = detect_sudden_changes_2d(action, **per_ep_kwargs)

    if startup_exclude_per_joint is not None:
        n = min(startup_exclude_per_joint.shape[1], state_flags.shape[1])
        apply_per_dim_exclude(state_flags[:, :n], startup_exclude_per_joint[:, :n])
        n_a = min(startup_exclude_per_joint.shape[1], action_flags.shape[1])
        apply_per_dim_exclude(action_flags[:, :n_a], startup_exclude_per_joint[:, :n_a])

    all_flags = np.concatenate([state_flags, action_flags], axis=1)

    if cfg.frame_abnormal_min_cluster > 1:
        for d in range(all_flags.shape[1]):
            all_flags[:, d] = _cluster_filter(all_flags[:, d], cfg.frame_abnormal_min_cluster)

    sudden_abnormal = all_flags.any(axis=1)
    abnormal |= sudden_abnormal
    flagged = int(all_flags.sum())

    state_dim = state.shape[1]
    for d in range(all_flags.shape[1]):
        col_flags = all_flags[:, d]
        if not col_flags.any():
            continue
        max_run = max_contiguous_run(col_flags)
        if max_run >= cfg.episode_min_cluster:
            padded = np.concatenate(([0], col_flags.astype(np.int8), [0]))
            diff = np.diff(padded)
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            signal = state[:, d] if d < state_dim else action[:, d - state_dim]
            for s, e in zip(starts, ends):
                if e - s < cfg.episode_min_cluster:
                    continue
                seg = signal[s:e]
                if seg.size >= 2 and float(np.abs(np.diff(seg)).max()) >= cfg.min_cluster_frame_jump:
                    discard = True
                    reasons.append(f"sudden_change_cluster_dim{d}_len{e - s}")
                    break

    return Stage1Result(
        discard=discard,
        discard_reasons=reasons,
        abnormal_frames=abnormal,
        flagged_count=flagged,
        hard_limit_hit=hard_hit,
    )
