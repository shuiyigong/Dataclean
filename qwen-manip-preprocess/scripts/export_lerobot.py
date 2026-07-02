#!/usr/bin/env python3
"""Export filtered pipeline output to a complete LeRobot v2.1 dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.lerobot_export import export_lerobot_dataset, verify_lerobot_alignment
from robot_data_processing.loader import list_episode_indices


def main(argv: list[str] | None = None) -> int:
    default_config_root = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged")
    parser = argparse.ArgumentParser(description="Export filtered data to LeRobot dataset format")
    parser.add_argument("--source-root", type=Path, default=default_config_root, help="Original LeRobot dataset root")
    parser.add_argument("--filtered-root", type=Path, required=True, help="Pipeline output dir with data_filtered/")
    parser.add_argument("--output-root", type=Path, required=True, help="Exported LeRobot dataset root")
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--episode-indices", type=str, default=None, help="Comma-separated episode indices")
    parser.add_argument("--skip-video-stats", action="store_true", help="Skip ffmpeg video stats sampling")
    parser.add_argument("--verify-only", action="store_true", help="Only run alignment verification")
    args = parser.parse_args(argv)

    if args.episode_indices:
        indices = [int(x.strip()) for x in args.episode_indices.split(",") if x.strip()]
    else:
        all_indices = list_episode_indices(args.source_root)
        limit = args.episode_limit or len(all_indices)
        indices = all_indices[:limit]

    if args.verify_only:
        report = verify_lerobot_alignment(args.output_root, indices)
        print(json.dumps(report, indent=2))
        return 0 if report["aligned"] else 1

    print(f"Exporting {len(indices)} episodes")
    print(f"  source:   {args.source_root}")
    print(f"  filtered: {args.filtered_root}")
    print(f"  output:   {args.output_root}")

    summary = export_lerobot_dataset(
        args.source_root,
        args.filtered_root,
        args.output_root,
        indices,
        recompute_video_stats=not args.skip_video_stats,
    )
    print(
        f"Exported {summary.total_episodes} episodes, "
        f"{summary.total_frames} frames, "
        f"{summary.truncated_episodes} truncated"
    )

    report = verify_lerobot_alignment(args.output_root, indices)
    report_path = args.output_root / "alignment_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2))
    print(f"Alignment report: {report_path}")
    return 0 if report["aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
