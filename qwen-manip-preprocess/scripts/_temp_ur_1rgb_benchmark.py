#!/usr/bin/env python3
"""TEMPORARY one-off runner for RoboMIND ur_1rgb (task-split LeRobot v3 layout).

Do not import from production pipeline modules beyond shared stage logic.
Dataset root is NOT modified; episodes are read from per-task file-*.parquet.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.mask import (
    build_frame_keep_mask,
    compute_per_joint_action_zero_exclude,
    compute_per_joint_stage1_exclude,
)
from robot_data_processing.report import build_quality_report, write_exclusion_log, write_quality_report
from robot_data_processing.schema import DatasetSchema
from robot_data_processing.stage1_stats import Stage1GlobalStats, load_or_compute_stage1_stats
from robot_data_processing.stages.stage1_sudden_change import Stage1Config, run_stage1
from robot_data_processing.stages.stage2_trend_alignment import Stage2Config, run_stage2
from robot_data_processing.stages.stage3_extreme_value import Stage3Config, run_stage3
from robot_data_processing.stages.stage4_static_interval import Stage4Config, run_stage4
from robot_data_processing.stages.state_action_temporal_alignment import (
    StateActionAlignConfig,
    StateActionAlignStats,
    aggregate_lag_stats,
    apply_state_action_temporal_alignment,
    compute_episode_lags,
    resolve_alignment_lag,
)
from robot_data_processing.stats import GlobalStats, load_or_compute_stats
from robot_data_processing.types import EpisodeResult

# --- temp schema (single UR arm: 7 joint+gripper state/action + 6 EE in state only) ---
TEMP_SCHEMA = DatasetSchema(
    embodiment="temp_ur_1rgb",
    layout="joint_gripper",
    state_dim=13,
    action_dim=7,
    alignment_dims=7,
    joint_index_list=tuple(range(6)),
    gripper_index_list=(6,),
    action_joint_index_list=tuple(range(6)),
    action_gripper_index_list=(6,),
    raw_columns=(),
)

DEFAULT_DATASET_ROOTS = [
    Path(
        "/mnt/project_rlinf_hs/data_move/16T_1_cjl_ysa/jlchen/datasets/"
        "RoboMIND_lerobot/benchmark1_0_compressed/ur_1rgb"
    ),
    Path(
        "/mnt/project_rlinf_hs/data_move/16T_1_cjl_ysa/jlchen/datasets/"
        "RoboMIND_lerobot/benchmark1_1_compressed/ur_1rgb"
    ),
]
DEFAULT_DATASET_ROOT = DEFAULT_DATASET_ROOTS[0]


@dataclass(frozen=True)
class EpisodeRef:
    global_index: int
    dataset_source: str
    task_name: str
    task_dir: Path
    data_path: Path
    episode_index: int


def discover_episodes(dataset_root: Path, source_name: str | None = None) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    source = source_name or dataset_root.parent.name
    gidx = 0
    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir():
            continue
        data_files = sorted((task_dir / "data").glob("chunk-*/file-*.parquet"))
        if not data_files:
            continue
        for data_path in data_files:
            table = pq.read_table(data_path, columns=["episode_index"])
            col = table.column("episode_index").combine_chunks()
            ep_ids = sorted({int(col[i].as_py()) for i in range(table.num_rows)})
            for ep in ep_ids:
                refs.append(
                    EpisodeRef(
                        global_index=gidx,
                        dataset_source=source,
                        task_name=task_dir.name,
                        task_dir=task_dir,
                        data_path=data_path,
                        episode_index=ep,
                    )
                )
                gidx += 1
    return refs


def discover_multi(dataset_roots: list[Path]) -> list[EpisodeRef]:
    all_refs: list[EpisodeRef] = []
    gidx = 0
    for root in dataset_roots:
        for ref in discover_episodes(root, source_name=root.parent.name):
            all_refs.append(
                EpisodeRef(
                    global_index=gidx,
                    dataset_source=ref.dataset_source,
                    task_name=ref.task_name,
                    task_dir=ref.task_dir,
                    data_path=ref.data_path,
                    episode_index=ref.episode_index,
                )
            )
            gidx += 1
    return all_refs


def read_episode_arrays(ref: EpisodeRef) -> dict[str, np.ndarray]:
    table = pq.read_table(
        ref.data_path,
        columns=[
            "observation.states.joint_position",
            "observation.states.end_effector",
            "actions.joint_position",
            "episode_index",
        ],
    )
    ep_col = table.column("episode_index").combine_chunks()
    keep = [i for i in range(table.num_rows) if int(ep_col[i].as_py()) == ref.episode_index]
    if not keep:
        return {"state": np.zeros((0, 13)), "action": np.zeros((0, 7))}
    sub = table.take(keep)
    joint = np.stack(sub.column("observation.states.joint_position").combine_chunks().to_pylist())
    ee = np.stack(sub.column("observation.states.end_effector").combine_chunks().to_pylist())
    action = np.stack(sub.column("actions.joint_position").combine_chunks().to_pylist())
    state = np.concatenate([joint, ee], axis=1).astype(np.float64)
    return {"state": state, "action": action.astype(np.float64)}


def _virtual_parquet_path(ref: EpisodeRef) -> Path:
    return ref.task_dir / f"__virtual_episode_{ref.episode_index:06d}.parquet"


# Patch helpers for stats loaders: map virtual path -> EpisodeRef via env dict
_REF_MAP: dict[str, EpisodeRef] = {}


def _install_ref_map(refs: list[EpisodeRef]) -> None:
    _REF_MAP.clear()
    for ref in refs:
        _REF_MAP[str(_virtual_parquet_path(ref))] = ref


def _patched_read_episode_canonical(path: Path, schema=TEMP_SCHEMA, action_from_state=False):
    ref = _REF_MAP[str(path)]
    data = read_episode_arrays(ref)
    return {"state": data["state"], "action": data["action"], "raw": {}}


def _patched_episode_parquet_path(root: Path, episode_index: int) -> Path:
    for ref in _REF_MAP.values():
        if ref.global_index == episode_index:
            return _virtual_parquet_path(ref)
    raise KeyError("unknown episode index " + str(episode_index))


def compute_lag_stats(refs: list[EpisodeRef], cfg: StateActionAlignConfig) -> StateActionAlignStats:
    per_ep: list[np.ndarray] = []
    for ref in refs:
        data = read_episode_arrays(ref)
        state, action = data["state"], data["action"]
        if state.shape[0] < cfg.min_active_samples + cfg.max_lag_frames + 1:
            continue
        per_ep.append(compute_episode_lags(state, action, cfg, TEMP_SCHEMA.alignment_dim))
    if not per_ep:
        z = np.full(TEMP_SCHEMA.alignment_dim, cfg.default_lag, dtype=np.float64)
        return StateActionAlignStats(
            lag_mean=cfg.default_lag,
            lag_min=cfg.default_lag,
            lag_max=cfg.default_lag,
            per_dim_lag_mean=z,
            per_dim_lag_min=z,
            per_dim_lag_max=z,
            num_episodes=0,
        )
    return aggregate_lag_stats(per_ep)


def process_one(
    ref: EpisodeRef,
    global_stats: GlobalStats,
    stage1_stats: Stage1GlobalStats,
    s1_cfg: Stage1Config,
    s2_cfg: Stage2Config,
    s3_cfg: Stage3Config,
    s4_cfg: Stage4Config,
    align_cfg: StateActionAlignConfig,
    align_lag: int,
) -> EpisodeResult:
    data = read_episode_arrays(ref)
    state, action = data["state"], data["action"]
    num_frames = state.shape[0]
    if num_frames == 0:
        return EpisodeResult(
            episode_index=ref.global_index,
            num_frames=0,
            discard=True,
            discard_reasons=["empty_episode"],
            step_validity_mask=np.array([], dtype=np.int8),
        )

    startup_exclude = compute_per_joint_action_zero_exclude(action, 1e-4)
    stage1_exclude = compute_per_joint_stage1_exclude(action, 1e-4, 60)

    s1 = run_stage1(
        state, action, s1_cfg, schema=TEMP_SCHEMA,
        startup_exclude_per_joint=stage1_exclude, global_stats=stage1_stats,
    )
    s2 = run_stage2(state, action, s2_cfg)
    s3 = run_stage3(
        state, action, global_stats, s3_cfg, schema=TEMP_SCHEMA,
        startup_exclude_per_joint=startup_exclude,
    )

    abnormal = s1.abnormal_frames.copy()
    reasons = list(s1.discard_reasons)
    if s2.discard:
        abnormal[:] = True
        reasons.extend(f"stage2:{r}" for r in s2.discard_reasons)
    else:
        abnormal |= s3.remove_frames
        reasons.extend(f"stage3:{r}" for r in s3.discard_reasons)

    valid_end = int(build_frame_keep_mask(num_frames, abnormal).sum()) if abnormal.any() else num_frames
    s4_remove = np.zeros(num_frames, dtype=bool)
    if valid_end > 0:
        s4 = run_stage4(state[:valid_end], action[:valid_end], s4_cfg)
        s4_remove[:valid_end] = s4.remove_frames
    else:
        s4 = run_stage4(state[:0], action[:0], s4_cfg)

    step_validity_mask = build_frame_keep_mask(num_frames, abnormal, s4_remove)
    kept_frames = int(step_validity_mask.sum())
    first_removed = int(np.flatnonzero(step_validity_mask == 0)[0]) if kept_frames < num_frames else None
    if first_removed is not None:
        reasons.append(f"first_removed_frame={first_removed}")

    aligned_action = apply_state_action_temporal_alignment(
        state, action, align_lag if align_cfg.enabled else 0, TEMP_SCHEMA.alignment_dim
    )
    _ = aligned_action  # post-stage alignment (not written back to dataset)

    return EpisodeResult(
        episode_index=ref.global_index,
        num_frames=num_frames,
        discard=False,
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
            "dataset_source": ref.dataset_source,
            "task_name": ref.task_name,
            "local_episode_index": ref.episode_index,
            "state_dim": 13,
            "action_dim": 7,
            "state_action_alignment_lag": align_lag if align_cfg.enabled else 0,
            "kept_frames": kept_frames,
        },
    )


def analyze_results(
    results: list[EpisodeResult],
    lag_stats,
    output_dir: Path,
    dataset_roots: list[Path] | None = None,
) -> dict:
    rows = [r.to_dict() for r in results]
    total_frames = sum(r["num_frames"] for r in rows)
    kept_frames = sum(r["kept_frames"] for r in rows)
    truncated = [r for r in rows if r["first_abnormal_frame"] is not None]
    da = [r["stage2_da_mean"] for r in rows if r["stage2_da_mean"] is not None]
    kept_ratios = [r["kept_frames"] / r["num_frames"] for r in rows if r["num_frames"] > 0]
    prefix_pcts = [
        100 * r["first_abnormal_frame"] / max(r["num_frames"] - 1, 1)
        for r in truncated if r["num_frames"] > 1
    ]
    task_trunc = {}
    source_stats: dict[str, dict] = {}
    for r in rows:
        src = r["metadata"].get("dataset_source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"episodes": 0, "frames": 0, "kept_frames": 0, "truncated": 0}
        source_stats[src]["episodes"] += 1
        source_stats[src]["frames"] += r["num_frames"]
        source_stats[src]["kept_frames"] += r["kept_frames"]
        if r["first_abnormal_frame"] is not None:
            source_stats[src]["truncated"] += 1
            t = r["metadata"]["task_name"]
            task_trunc[f"{src}/{t}"] = task_trunc.get(f"{src}/{t}", 0) + 1

    for src, st in source_stats.items():
        st["valid_frame_rate_pct"] = round(100 * st["kept_frames"] / max(st["frames"], 1), 2)
        st["truncation_rate_pct"] = round(100 * st["truncated"] / max(st["episodes"], 1), 2)

    return {
        "dataset_roots": [str(p) for p in dataset_roots] if dataset_roots else [],
        "layout_note": "task subdirs per benchmark, LeRobot v3 file-*.parquet, single UR arm",
        "full_run": dataset_roots is not None and len(results) > 2000,
        "sample_size": len(rows),
        "total_frames": total_frames,
        "kept_frames": kept_frames,
        "valid_frame_rate_pct": round(100 * kept_frames / max(total_frames, 1), 2),
        "episodes_fully_kept": len(rows) - len(truncated),
        "episodes_prefix_truncated": len(truncated),
        "truncation_rate_pct": round(100 * len(truncated) / max(len(rows), 1), 2),
        "kept_ratio_per_episode": {
            "mean": round(float(np.mean(kept_ratios)), 4),
            "median": round(float(np.median(kept_ratios)), 4),
            "p10": round(float(np.percentile(kept_ratios, 10)), 4),
            "min": round(float(np.min(kept_ratios)), 4),
        },
        "first_abnormal_position_pct": {
            "mean": round(float(np.mean(prefix_pcts)), 2) if prefix_pcts else None,
            "median": round(float(np.median(prefix_pcts)), 2) if prefix_pcts else None,
        },
        "stage2_da_mean": {
            "mean": round(float(np.mean(da)), 4),
            "median": round(float(np.median(da)), 4),
            "min": round(float(np.min(da)), 4),
            "below_0.65": sum(1 for x in da if x < 0.65),
        },
        "stage2_discarded_episodes": sum(
            1 for r in rows if any(x.startswith("stage2:") for x in r.get("discard_reasons", []))
        ),
        "stage3_excluded_frames_total": sum(r.get("stage3_excluded_frames", 0) for r in rows),
        "stage4_removed_frames_total": sum(r.get("stage4_removed_frames", 0) for r in rows),
        "state_action_alignment": {
            "global_lag_mean": lag_stats.lag_mean,
            "global_lag_min": lag_stats.lag_min,
            "global_lag_max": lag_stats.lag_max,
            "per_dim_lag_mean": lag_stats.per_dim_lag_mean.round(2).tolist(),
        },
        "top_tasks_by_truncation": dict(sorted(task_trunc.items(), key=lambda x: -x[1])[:15]),
        "per_dataset_source": source_stats,
    }


def main() -> None:
    import argparse
    import robot_data_processing.loader as loader_mod
    import robot_data_processing.stats as stats_mod
    import robot_data_processing.stage1_stats as s1stats_mod

    parser = argparse.ArgumentParser(description="TEMP ur_1rgb benchmark runner")
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        type=Path,
        default=DEFAULT_DATASET_ROOTS,
        help="One or more ur_1rgb dataset roots",
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Deprecated: single root")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "ur_1rgb_full")
    parser.add_argument("--sample-size", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()

    dataset_roots = [args.dataset_root] if args.dataset_root else list(args.dataset_roots)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Discovering episodes...")
    all_refs = discover_multi(dataset_roots)
    per_src = {}
    for ref in all_refs:
        per_src[ref.dataset_source] = per_src.get(ref.dataset_source, 0) + 1
    print(f"Total episodes: {len(all_refs)} ({per_src})")

    if args.sample_size <= 0:
        refs = all_refs
        for i, ref in enumerate(refs):
            refs[i] = EpisodeRef(
                global_index=i,
                dataset_source=ref.dataset_source,
                task_name=ref.task_name,
                task_dir=ref.task_dir,
                data_path=ref.data_path,
                episode_index=ref.episode_index,
            )
        n = len(refs)
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.sample_size, len(all_refs))
        pick = sorted(rng.choice(len(all_refs), size=n, replace=False).tolist())
        refs = [all_refs[i] for i in pick]
        for i, ref in enumerate(refs):
            refs[i] = EpisodeRef(
                global_index=i,
                dataset_source=ref.dataset_source,
                task_name=ref.task_name,
                task_dir=ref.task_dir,
                data_path=ref.data_path,
                episode_index=ref.episode_index,
            )
    (out / "episode_refs.json").write_text(
        json.dumps(
            [
                ref.__dict__ | {"task_dir": str(ref.task_dir), "data_path": str(ref.data_path)}
                for ref in refs
            ],
            indent=2,
        )
    )

    _install_ref_map(refs)
    loader_mod.read_episode_canonical = _patched_read_episode_canonical
    loader_mod.episode_parquet_path = _patched_episode_parquet_path
    stats_mod.read_episode_canonical = _patched_read_episode_canonical
    stats_mod.episode_parquet_path = _patched_episode_parquet_path
    s1stats_mod.read_episode_canonical = _patched_read_episode_canonical
    s1stats_mod.episode_parquet_path = _patched_episode_parquet_path

    virtual_root = out / "_virtual_root"
    virtual_root.mkdir(exist_ok=True)
    ep_indices = [r.global_index for r in refs]

    s1_cfg = Stage1Config(
        k_residual=60, k_accel=60, k_jerk=60, percentile_floor=99.999,
        joint_abs_max=4.0, ee_position_max=4.0, gripper_max=1.0,
        frame_max_cluster=4, episode_min_cluster=50, on_hard_limit_violation=True,
    )
    s2_cfg = Stage2Config(da_per_dim=0.65, da_episode_mean=0.65)
    s3_cfg = Stage3Config(
        alpha=0.35,
        gripper_state_indices=(6,), gripper_action_indices=(6,),
        joint_limits=(-4.0, 4.0), ee_xyz_limits=(-4.0, 4.0),
        gripper_limits=(-0.01, 1.0),
    )
    s4_cfg = Stage4Config(max_static_steps=5)
    align_cfg = StateActionAlignConfig(enabled=True, default_lag=1)

    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    print("Computing global stats...")
    global_stats = load_or_compute_stats(
        virtual_root, cache / "global_stats.npz", ep_indices,
        schema=TEMP_SCHEMA, recompute=True, num_workers=args.num_workers,
        num_bins=65536,
    )
    print("Computing stage1 stats...")
    stage1_stats = load_or_compute_stage1_stats(
        virtual_root, cache / "stage1_global_stats.npz", ep_indices,
        TEMP_SCHEMA, s1_cfg, recompute=True, num_workers=args.num_workers,
        num_bins=65536,
    )
    print("Computing state-action lag stats...")
    lag_stats = compute_lag_stats(refs, align_cfg)
    align_lag = resolve_alignment_lag(align_cfg, lag_stats)
    print(f"Alignment lag: {align_lag}")

    print("Processing episodes...")
    results: list[EpisodeResult] = []
    for ref in tqdm(refs, desc="process"):
        results.append(
            process_one(ref, global_stats, stage1_stats, s1_cfg, s2_cfg, s3_cfg, s4_cfg, align_cfg, align_lag)
        )

    report = build_quality_report(
        results,
        stats_meta={"num_episodes": len(results), "embodiment": "temp_ur_1rgb", "alignment_lag": align_lag},
        config_summary={
            "sample_size": n,
            "full_run": args.sample_size <= 0,
            "dataset_roots": [str(p) for p in dataset_roots],
            "seed": args.seed,
            "num_workers": args.num_workers,
        },
    )
    write_quality_report(out / "reports" / "quality_report.json", report)
    write_exclusion_log(out / "reports" / "exclusion_log.jsonl", results)

    analysis = analyze_results(results, lag_stats, out, dataset_roots=dataset_roots)
    (out / "analysis" / "analysis_report.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "analysis_report.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
