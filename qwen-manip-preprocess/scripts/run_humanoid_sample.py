#!/usr/bin/env python3
"""Run humanoid_merged pipeline on a random episode sample."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.loader import list_episode_indices
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "humanoid_merged.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--output-mode", choices=["report", "filter", "both"], default="report")
    parser.add_argument("--enable-stage5", action="store_true", help="Run stage5 (needs calibration files)")
    parser.add_argument("--stage2-diff-epsilon", type=float, default=None)
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    dataset_root = Path(yaml_cfg["dataset"]["root"])
    total = yaml_cfg["dataset"]["total_episodes"]
    all_indices = list_episode_indices(dataset_root, total)
    print(f"Dataset episodes available: {len(all_indices)}")

    rng = np.random.default_rng(args.seed)
    n = min(args.sample_size, len(all_indices))
    sample = sorted(rng.choice(all_indices, size=n, replace=False).tolist())

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "sampled_episode_indices.json").write_text(json.dumps(sample, indent=2))

    stage5_enabled = bool(args.enable_stage5)
    overrides = {
        "output_dir": str(out),
        "output_mode": args.output_mode,
        "num_workers": args.num_workers,
        "stage5_enabled": stage5_enabled,
    }
    if args.stage2_diff_epsilon is not None:
        overrides["stage2_diff_epsilon"] = args.stage2_diff_epsilon
    cfg = pipeline_config_from_yaml(yaml_cfg, overrides)

    print(f"Sample size: {n}, seed: {args.seed}")
    print(f"Output: {out}, mode: {args.output_mode}, workers: {args.num_workers}, stage5={stage5_enabled}")
    if args.stage2_diff_epsilon is not None:
        print(f"Stage2 diff_epsilon: {args.stage2_diff_epsilon}")

    results = run_pipeline(cfg, sample, stats_episode_indices=sample)
    discarded = sum(1 for r in results if r.discard)
    kept_frames = sum(r.kept_frames for r in results)
    total_frames = sum(r.num_frames for r in results)
    print("\n=== Done ===")
    print(f"Processed: {len(results)}")
    print(f"Discarded: {discarded} ({100*discarded/len(results):.2f}%)")
    print(f"Kept frames: {kept_frames}/{total_frames} ({100*kept_frames/max(total_frames,1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
