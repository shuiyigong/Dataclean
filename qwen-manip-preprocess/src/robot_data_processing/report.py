from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from robot_data_processing.types import EpisodeResult


def write_exclusion_log(path: Path, results: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def build_quality_report(
    results: list[EpisodeResult],
    stats_meta: dict,
    config_summary: dict,
) -> dict:
    total = len(results)
    discarded = [r for r in results if r.discard]
    kept = [r for r in results if not r.discard]

    reason_counter: Counter = Counter()
    for r in discarded:
        for reason in r.discard_reasons:
            reason_counter[reason.split("=")[0]] += 1

    total_frames = sum(r.num_frames for r in results)
    kept_frames = sum(r.kept_frames for r in results)
    stage1_flagged = sum(r.stage1_flagged_frames for r in results)
    stage3_excluded = sum(r.stage3_excluded_frames for r in results)
    stage4_removed = sum(r.stage4_removed_frames for r in results)

    da_values = [r.stage2_da_mean for r in results if r.stage2_da_mean is not None]
    low_prefix = sum(1 for r in results if r.metadata.get("low_valid_prefix"))

    report = {
        "summary": {
            "total_episodes": total,
            "discarded_episodes": len(discarded),
            "kept_episodes": len(kept),
            "discard_rate": len(discarded) / total if total else 0.0,
            "low_valid_prefix_episodes": low_prefix,
            "total_frames": total_frames,
            "kept_frames": kept_frames,
            "valid_prefix_frame_rate": kept_frames / total_frames if total_frames else 0.0,
            "stage1_flagged_frames": stage1_flagged,
            "stage3_excluded_frames": stage3_excluded,
            "stage4_removed_frames": stage4_removed,
        },
        "stage2_da": {
            "mean": float(sum(da_values) / len(da_values)) if da_values else None,
            "min": float(min(da_values)) if da_values else None,
            "max": float(max(da_values)) if da_values else None,
            "below_0.7": sum(1 for v in da_values if v < 0.7),
        },
        "discard_reasons": dict(reason_counter),
        "stats_meta": stats_meta,
        "config": config_summary,
    }
    return report


def write_quality_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
