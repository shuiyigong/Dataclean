#!/usr/bin/env python3
"""Materialize action=state preprocessing into parquet files on disk."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.loader import episode_parquet_path, list_episode_indices, read_episode_table
from robot_data_processing.preprocess import replace_action_with_state_table


def _process_episode(source_root: Path, output_root: Path, episode_index: int) -> bool:
    src = episode_parquet_path(source_root, episode_index)
    if not src.exists():
        return False
    chunk = episode_index // 1000
    dst = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    table = read_episode_table(src)
    table = replace_action_with_state_table(table)
    pq.write_table(table, dst, compression="snappy")
    return True


def main(argv: list[str] | None = None) -> int:
    default_root = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged")
    parser = argparse.ArgumentParser(description="Write action=state preprocessed parquets")
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args(argv)

    indices = list_episode_indices(args.source_root)
    if args.episode_limit is not None:
        indices = indices[: args.episode_limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = partial(_process_episode, args.source_root, args.output_root)
    ctx = mp.get_context("fork")
    ok = 0
    with ctx.Pool(processes=args.num_workers) as pool:
        for success in tqdm(pool.imap_unordered(worker, indices, chunksize=8), total=len(indices)):
            ok += int(success)

    print(f"Wrote {ok}/{len(indices)} episodes to {args.output_root / 'data'}")
    print("Point pipeline --dataset-root to output-root (with videos/meta symlinked or copied separately).")
    return 0 if ok == len(indices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
