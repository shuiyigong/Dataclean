from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_data_processing.loader import (
    episode_parquet_path,
    read_episode_canonical,
    read_episode_table,
    write_episode_with_validity_mask,
)
from robot_data_processing.preprocess import replace_action_with_state_table
from robot_data_processing.mask import (
    build_frame_keep_mask,
    compute_per_joint_action_zero_exclude,
    compute_per_joint_stage1_exclude,
)
from robot_data_processing.report import build_quality_report, write_exclusion_log, write_quality_report
from robot_data_processing.schema import DatasetSchema, schema_from_yaml
from robot_data_processing.stages.stage1_sudden_change import Stage1Config, run_stage1
from robot_data_processing.stages.stage2_trend_alignment import Stage2Config, run_stage2
from robot_data_processing.stages.stage3_extreme_value import Stage3Config, run_stage3
from robot_data_processing.stages.stage4_static_interval import Stage4Config, run_stage4
from robot_data_processing.stages.stage5_frame_alignment import (
    Stage5Config,
    read_stage5_raw_context,
    run_stage5,
)
from robot_data_processing.stage1_stats import Stage1GlobalStats, load_or_compute_stage1_stats
from robot_data_processing.stats import load_or_compute_stats
from robot_data_processing.types import EpisodeResult, GlobalStats


@dataclass
class PipelineConfig:
    dataset_root: Path
    output_dir: Path
    output_mode: str
    schema: DatasetSchema
    num_workers: int = 128
    min_episode_length: int = 30
    stage1: Stage1Config | None = None
    stage2: Stage2Config | None = None
    stage3: Stage3Config | None = None
    stage4: Stage4Config | None = None
    stage5: Stage5Config | None = None
    stats_recompute: bool = True
    stats_cache_path: Path | None = None
    stage1_stats_recompute: bool = True
    stage1_stats_cache_path: Path | None = None
    stats_num_bins: int = 65536
    total_episodes: int | None = None
    action_zero_epsilon: float = 1e-4
    stage1_post_zero_grace_frames: int = 0
    discard_short_prefix: bool = False
    action_from_state: bool = False


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(global_stats: GlobalStats, stage1_stats: Stage1GlobalStats, pipe_cfg: PipelineConfig) -> None:
    _WORKER_STATE["stats"] = global_stats
    _WORKER_STATE["stage1_stats"] = stage1_stats
    _WORKER_STATE["cfg"] = pipe_cfg


def process_episode(episode_index: int, pipe_cfg: PipelineConfig | None = None) -> EpisodeResult:
    cfg = pipe_cfg or _WORKER_STATE["cfg"]
    stats: GlobalStats = _WORKER_STATE.get("stats")  # type: ignore[assignment]
    stage1_stats: Stage1GlobalStats = _WORKER_STATE.get("stage1_stats")  # type: ignore[assignment]
    schema = cfg.schema
    if stats is None and cfg is not None:
        raise RuntimeError("Global stats not initialized in worker")
    if stage1_stats is None and cfg is not None:
        raise RuntimeError("Stage1 global stats not initialized in worker")

    path = episode_parquet_path(cfg.dataset_root, episode_index)
    if not path.exists():
        return EpisodeResult(
            episode_index=episode_index,
            num_frames=0,
            discard=True,
            discard_reasons=["missing_parquet"],
            step_validity_mask=np.array([], dtype=np.int8),
        )

    data = read_episode_canonical(path, schema=schema, action_from_state=cfg.action_from_state)
    state = data["state"]
    action = data["action"]
    num_frames = state.shape[0]

    s1_cfg = cfg.stage1 or Stage1Config()
    s2_cfg = cfg.stage2 or Stage2Config()
    s3_cfg = cfg.stage3 or Stage3Config()
    s4_cfg = cfg.stage4 or Stage4Config()
    s5_cfg = cfg.stage5 or Stage5Config()

    startup_exclude = compute_per_joint_action_zero_exclude(action, cfg.action_zero_epsilon)
    stage1_exclude = compute_per_joint_stage1_exclude(
        action, cfg.action_zero_epsilon, cfg.stage1_post_zero_grace_frames
    )

    s1 = run_stage1(
        state,
        action,
        s1_cfg,
        schema=schema,
        startup_exclude_per_joint=stage1_exclude,
        global_stats=stage1_stats,
    )
    s2 = run_stage2(state, action, s2_cfg)
    s3 = run_stage3(
        state, action, stats, s3_cfg, schema=schema, startup_exclude_per_joint=startup_exclude
    )

    abnormal = s1.abnormal_frames.copy()
    reasons: list[str] = list(s1.discard_reasons)

    if s2.discard:
        abnormal[:] = True
        reasons.extend(f"stage2:{r}" for r in s2.discard_reasons)
    else:
        abnormal |= s3.remove_frames
        reasons.extend(f"stage3:{r}" for r in s3.discard_reasons)

    prefix_mask = build_frame_keep_mask(num_frames, abnormal)
    valid_end = int(prefix_mask.sum()) if prefix_mask.any() else 0
    s4_remove = np.zeros(num_frames, dtype=bool)
    if valid_end > 0:
        s4 = run_stage4(state[:valid_end], action[:valid_end], s4_cfg)
        s4_remove[:valid_end] = s4.remove_frames
    else:
        s4 = run_stage4(state[:0], action[:0], s4_cfg)
    step_validity_mask = build_frame_keep_mask(num_frames, abnormal, s4_remove)
    kept_frames = int(step_validity_mask.sum())
    first_removed = int(np.flatnonzero(step_validity_mask == 0)[0]) if kept_frames < num_frames else None

    discard = False
    if cfg.discard_short_prefix and kept_frames < cfg.min_episode_length:
        discard = True
        reasons.append(f"kept_frames={kept_frames}<{cfg.min_episode_length}")
    if first_removed is not None:
        reasons.append(f"first_removed_frame={first_removed}")

    s5_meta: dict[str, Any] = {"stage5_applied": False}
    if s5_cfg.enabled:
        raw_ctx = read_stage5_raw_context(path, schema, s5_cfg)
        s5 = run_stage5(cfg.dataset_root, episode_index, schema, raw_ctx, s5_cfg)
        s5_meta = {
            "stage5_applied": s5.applied,
            "stage5_reference_frame": s5_cfg.reference_frame,
            "stage5_reference_forward": s5.reference_forward_xyz.tolist(),
            "stage5_rotation_correction_euler_xyz": list(s5_cfg.rotation_correction_euler_xyz),
            "aligned_state_shape": list(s5.aligned_state.shape),
        }

    return EpisodeResult(
        episode_index=episode_index,
        num_frames=num_frames,
        discard=discard,
        discard_reasons=reasons,
        step_validity_mask=step_validity_mask,
        first_abnormal_frame=first_removed,
        stage1_flagged_frames=s1.flagged_count,
        stage2_da_mean=s2.da_mean,
        stage2_da_per_dim=s2.da_per_dim,
        stage2_lags=s2.lags,
        stage3_excluded_frames=s3.excluded_count,
        stage4_removed_frames=s4.removed_count,
        metadata={
            "stage1_hard_limit": s1.hard_limit_hit,
            "valid_prefix": kept_frames,
            "kept_frames": kept_frames,
            "low_valid_prefix": kept_frames < cfg.min_episode_length,
            "stage4_static_runs_shortened": s4.static_runs_shortened,
            "stage4_static_frames_removed": s4.static_frames_removed,
            "startup_exclude_per_joint_frames": int(startup_exclude.sum()),
            "stage1_exclude_frames": int(stage1_exclude.sum()),
            "stage1_post_zero_grace_frames": cfg.stage1_post_zero_grace_frames,
            "embodiment": schema.embodiment,
            "canonical_dim": schema.canonical_dim,
            **s5_meta,
        },
    )


def _worker_fn(episode_index: int) -> EpisodeResult:
    return process_episode(episode_index)


def _write_output_episode(cfg: PipelineConfig, result: EpisodeResult) -> None:
    src = episode_parquet_path(cfg.dataset_root, result.episode_index)
    chunk = result.episode_index // 1000
    dst = cfg.output_dir / "data_filtered" / f"chunk-{chunk:03d}" / f"episode_{result.episode_index:06d}.parquet"
    table = read_episode_table(src)
    if cfg.action_from_state:
        table = replace_action_with_state_table(table, schema=cfg.schema)
    write_episode_with_validity_mask(dst, table, result.step_validity_mask)

    s5_cfg = cfg.stage5 or Stage5Config()
    if not s5_cfg.enabled:
        return
    path = episode_parquet_path(cfg.dataset_root, result.episode_index)
    raw_ctx = read_stage5_raw_context(path, cfg.schema, s5_cfg)
    s5 = run_stage5(cfg.dataset_root, result.episode_index, cfg.schema, raw_ctx, s5_cfg)
    aligned_dir = cfg.output_dir / "data_aligned" / f"chunk-{chunk:03d}"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    mask = result.step_validity_mask if result.step_validity_mask is not None else np.ones(s5.aligned_state.shape[0], dtype=np.int8)
    np.savez_compressed(
        aligned_dir / f"episode_{result.episode_index:06d}.npz",
        aligned_state=s5.aligned_state.astype(np.float32),
        aligned_action=s5.aligned_action.astype(np.float32),
        step_validity_mask=mask.astype(np.int8),
        reference_frame=s5_cfg.reference_frame,
        rotation_correction_euler_xyz=np.array(s5_cfg.rotation_correction_euler_xyz, dtype=np.float64),
    )


def run_pipeline(
    cfg: PipelineConfig,
    episode_indices: list[int],
    stats_episode_indices: list[int] | None = None,
    show_progress: bool = True,
) -> list[EpisodeResult]:
    from tqdm import tqdm

    stats_eps = stats_episode_indices or episode_indices
    cache_path = cfg.stats_cache_path or (cfg.output_dir / "cache" / "global_stats.npz")

    global_stats = load_or_compute_stats(
        cfg.dataset_root,
        cache_path,
        stats_eps,
        schema=cfg.schema,
        recompute=cfg.stats_recompute,
        num_workers=cfg.num_workers,
        num_bins=cfg.stats_num_bins,
        show_progress=show_progress,
        action_from_state=cfg.action_from_state,
    )

    s1_cache = cfg.stage1_stats_cache_path or (cfg.output_dir / "cache" / "stage1_global_stats.npz")
    s1_cfg = cfg.stage1 or Stage1Config()
    stage1_global_stats = load_or_compute_stage1_stats(
        cfg.dataset_root,
        s1_cache,
        stats_eps,
        cfg.schema,
        s1_cfg,
        recompute=cfg.stage1_stats_recompute,
        num_workers=cfg.num_workers,
        num_bins=cfg.stats_num_bins,
        show_progress=show_progress,
        action_from_state=cfg.action_from_state,
    )

    ctx = mp.get_context("fork")
    results: list[EpisodeResult] = []
    with ctx.Pool(
        processes=cfg.num_workers,
        initializer=_init_worker,
        initargs=(global_stats, stage1_global_stats, cfg),
    ) as pool:
        iterator = pool.imap_unordered(_worker_fn, episode_indices, chunksize=4)
        if show_progress:
            iterator = tqdm(iterator, total=len(episode_indices), desc="process episodes")
        for result in iterator:
            results.append(result)
            if cfg.output_mode in ("filter", "both") and result.step_validity_mask is not None:
                if result.step_validity_mask.size > 0:
                    _write_output_episode(cfg, result)

    results.sort(key=lambda r: r.episode_index)

    if cfg.output_mode in ("report", "both"):
        report = build_quality_report(
            results,
            stats_meta={
                "num_frames": global_stats.num_frames,
                "num_episodes": global_stats.num_episodes,
                "cache_path": str(cache_path),
                "stage1_stats_cache_path": str(s1_cache),
                "stage1_threshold_mode": "global",
                "embodiment": cfg.schema.embodiment,
                "canonical_dim": cfg.schema.canonical_dim,
            },
            config_summary={
                "output_mode": cfg.output_mode,
                "num_workers": cfg.num_workers,
                "processed_episodes": len(episode_indices),
                "validity_mask_mode": "prefix_truncate_then_static_shorten",
                "action_from_state": cfg.action_from_state,
                "stage4_max_static_steps": (cfg.stage4 or Stage4Config()).max_static_steps,
                "stage5_enabled": (cfg.stage5 or Stage5Config()).enabled,
                "embodiment": cfg.schema.embodiment,
            },
        )
        write_quality_report(cfg.output_dir / "reports" / "quality_report.json", report)
        write_exclusion_log(cfg.output_dir / "reports" / "exclusion_log.jsonl", results)

    return results


def load_config(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pipeline_config_from_yaml(yaml_cfg: dict, overrides: dict | None = None) -> PipelineConfig:
    overrides = overrides or {}
    dataset_root = Path(overrides.get("dataset_root", yaml_cfg["dataset"]["root"]))
    output_dir = Path(overrides.get("output_dir", "./output"))
    schema = schema_from_yaml(yaml_cfg)
    s1 = yaml_cfg["stage1"]
    s2 = yaml_cfg["stage2"]
    s3 = yaml_cfg["stage3"]
    pipe = yaml_cfg.get("pipeline", {})
    s4 = yaml_cfg.get("stage4", {})
    s5 = yaml_cfg.get("stage5", {})
    preprocess = yaml_cfg.get("preprocess", {})

    gripper_state = tuple(s3["exempt_dims"].get("gripper_state", list(schema.gripper_indices)))
    gripper_action = tuple(s3["exempt_dims"].get("gripper_action", list(schema.gripper_indices)))
    rpy_state = tuple(s3["exempt_dims"].get("rpy_state", list(schema.rpy_indices)))
    rpy_action = tuple(s3["exempt_dims"].get("rpy_action", list(schema.rpy_indices)))

    return PipelineConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        output_mode=overrides.get("output_mode", pipe.get("output_mode", "report")),
        schema=schema,
        num_workers=int(overrides.get("num_workers", pipe.get("num_workers", 128))),
        min_episode_length=int(overrides.get("min_episode_length", pipe.get("min_episode_length", 30))),
        action_zero_epsilon=float(overrides.get("action_zero_epsilon", pipe.get("action_zero_epsilon", 1e-4))),
        stage1_post_zero_grace_frames=int(
            overrides.get("stage1_post_zero_grace_frames", pipe.get("stage1_post_zero_grace_frames", 0))
        ),
        discard_short_prefix=bool(overrides.get("discard_short_prefix", pipe.get("discard_short_prefix", False))),
        action_from_state=bool(overrides.get("action_from_state", preprocess.get("action_from_state", False))),
        stage1=Stage1Config(
            median_kernel=s1["smoothing"]["median_kernel"],
            savgol_window=s1["smoothing"]["savgol_window"],
            savgol_polyorder=s1["smoothing"]["savgol_polyorder"],
            k_residual=s1["thresholds"]["k_residual"],
            k_accel=s1["thresholds"]["k_accel"],
            k_jerk=s1["thresholds"]["k_jerk"],
            percentile_floor=s1["thresholds"]["percentile_floor"],
            joint_abs_max=s1["hard_limits"]["joint_abs_max"],
            ee_position_max=s1["hard_limits"]["ee_position_max"],
            rpy_abs_max=s1["hard_limits"].get("rpy_abs_max", 3.14159265),
            gripper_max=s1["hard_limits"]["gripper_max"],
            frame_max_cluster=s1["exclusion"]["frame_removal"]["max_cluster_length"],
            frame_abnormal_min_cluster=s1["exclusion"]["frame_removal"].get(
                "min_cluster_length_for_abnormal", 1
            ),
            episode_min_cluster=s1["exclusion"]["episode_discard"]["min_cluster_length"],
            min_cluster_frame_jump=s1["exclusion"]["episode_discard"].get("min_cluster_frame_jump", 0.08),
            on_hard_limit_violation=s1["exclusion"]["episode_discard"]["on_hard_limit_violation"],
        ),
        stage2=Stage2Config(
            median_kernel=s2["smoothing"]["median_kernel"],
            savgol_window=s2["smoothing"]["savgol_window"],
            savgol_polyorder=s2["smoothing"]["savgol_polyorder"],
            max_lag_frames=s2["alignment"]["max_lag_frames"],
            diff_epsilon=s2["alignment"]["diff_epsilon"],
            min_active_samples=s2["alignment"]["min_active_samples"],
            da_per_dim=s2["thresholds"]["da_per_dim"],
            da_episode_mean=s2["thresholds"]["da_episode_mean"],
            action_type=s2["action_type"],
        ),
        stage3=Stage3Config(
            alpha=s3["alpha"],
            gripper_state_indices=gripper_state,
            gripper_action_indices=gripper_action,
            rpy_state_indices=rpy_state,
            rpy_action_indices=rpy_action,
            joint_limits=tuple(s3["hard_limits"]["joint"]),
            ee_xyz_limits=tuple(s3["hard_limits"]["ee_xyz"]),
            rpy_limits=tuple(s3["hard_limits"].get("rpy", [-3.15, 3.15])),
            gripper_limits=tuple(s3["hard_limits"]["gripper"]),
            min_episode_length=s3["exclusion"]["min_episode_length"],
        ),
        stage4=Stage4Config(
            max_static_steps=int(overrides.get("stage4_max_static_steps", s4.get("max_static_steps", 5))),
            enabled=bool(overrides.get("stage4_enabled", s4.get("enabled", True))),
            change_epsilon=float(overrides.get("stage4_change_epsilon", s4.get("change_epsilon", 0.0))),
        ),
        stage5=Stage5Config(
            enabled=bool(overrides.get("stage5_enabled", s5.get("enabled", False))),
            reference_frame=s5.get("reference_frame", "camera_top_frame0"),
            rotation_correction_euler_xyz=tuple(
                s5.get("rotation_correction_euler_xyz", [0.0, 0.0, 0.0])
            ),
            egodex_extrinsics_column=s5.get(
                "egodex_extrinsics_column", "observation.camera_extrinsics_world"
            ),
            humanoid_calibration_relpath=s5.get(
                "humanoid_calibration_relpath",
                "parameters/chunk-{chunk:03d}/episode_{episode:06d}/calibration_bundle_optimized.json",
            ),
            humanoid_camera_extrinsic_key=s5.get(
                "humanoid_camera_extrinsic_key", "camera_front_to_arm_left"
            ),
        ),
        stats_recompute=bool(overrides.get("stats_recompute", s3["stats"]["recompute"])),
        stats_cache_path=Path(overrides.get("stats_cache_path", output_dir / s3["stats"]["cache_path"])),
        stage1_stats_recompute=bool(overrides.get("stage1_stats_recompute", s1["stats"].get("recompute", True))),
        stage1_stats_cache_path=Path(
            overrides.get("stage1_stats_cache_path", output_dir / s1["stats"].get("cache_path", "cache/stage1_global_stats.npz"))
        ),
        stats_num_bins=int(s3["stats"]["num_histogram_bins"]),
        total_episodes=yaml_cfg["dataset"].get("total_episodes"),
    )
