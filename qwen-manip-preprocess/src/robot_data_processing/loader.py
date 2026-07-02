from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_processing.preprocess import replace_action_with_state
from robot_data_processing.schema import (
    DatasetSchema,
    EGODEX_SCHEMA,
    HUMANOID_SCHEMA,
    stack_column,
)
from robot_data_processing.transforms import (
    egodex_to_canonical,
    humanoid_action_to_canonical,
    humanoid_state_to_canonical,
)


def episode_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def list_episode_indices(root: Path, total_episodes: int | None = None) -> list[int]:
    indices: list[int] = []
    if total_episodes is not None:
        for i in range(total_episodes):
            if episode_parquet_path(root, i).exists():
                indices.append(i)
        return indices

    data_root = root / "data"
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for path in sorted(chunk_dir.glob("episode_*.parquet")):
            indices.append(int(path.stem.split("_")[1]))
    return sorted(indices)


def _transform_to_canonical(
    raw: dict[str, np.ndarray],
    schema: DatasetSchema,
) -> tuple[np.ndarray, np.ndarray]:
    if schema.embodiment == "egodex":
        state = egodex_to_canonical(raw[schema.state_column])
        action = egodex_to_canonical(raw[schema.action_column])
        return state, action
    if schema.embodiment == "humanoid":
        return (
            humanoid_state_to_canonical(raw[schema.state_column]),
            humanoid_action_to_canonical(raw[schema.action_column]),
        )
    raise ValueError(f"Unsupported embodiment: {schema.embodiment}")


def read_episode_canonical(
    path: Path,
    schema: DatasetSchema = HUMANOID_SCHEMA,
    action_from_state: bool = False,
) -> dict[str, np.ndarray]:
    """Read parquet and return canonical state/action (T, 14) arrays."""
    table = pq.read_table(path, columns=list(schema.raw_columns))
    raw: dict[str, np.ndarray] = {}
    for name in schema.raw_columns:
        if name not in table.column_names:
            continue
        col = table.column(name)
        if hasattr(col, "combine_chunks"):
            col = col.combine_chunks()
        raw[name] = stack_column(col.to_numpy(zero_copy_only=False))

    if action_from_state:
        replace_action_with_state(raw, schema)

    state, action = _transform_to_canonical(raw, schema)
    return {"state": state, "action": action, "raw": raw}


def read_episode_arrays(
    path: Path,
    columns: tuple[str, ...] | None = None,
    action_from_state: bool = False,
    schema: DatasetSchema = HUMANOID_SCHEMA,
) -> dict[str, np.ndarray]:
    """Backward-compatible read; returns canonical state/action plus legacy keys when available."""
    data = read_episode_canonical(path, schema=schema, action_from_state=action_from_state)
    out: dict[str, np.ndarray] = {
        "state": data["state"],
        "action": data["action"],
        "observation.state": data["state"],
    }
    raw = data.get("raw", {})
    for key in (
        "observation.state.arm.position",
        "observation.state.end.position",
        "action.arm.position",
    ):
        if key in raw:
            out[key] = raw[key]
    if schema.embodiment == "humanoid" and "observation.state" in raw:
        out["observation.state"] = raw["observation.state"]
        if "observation.state.arm.position" in raw:
            out["observation.state.arm.position"] = raw["observation.state.arm.position"]
        if "observation.state.end.position" in raw:
            out["observation.state.end.position"] = raw["observation.state.end.position"]
        if "action.arm.position" in raw:
            out["action.arm.position"] = raw["action.arm.position"]
    return out


def read_episode_table(path: Path):
    """Read full parquet table for output with validity mask."""
    return pq.read_table(path)


def valid_keep_length(table) -> int:
    """Number of frames marked keep (mask==1); supports prefix-only or sparse masks."""
    if "step_validity_mask" not in table.column_names:
        return table.num_rows
    col = table.column("step_validity_mask").combine_chunks()
    count = 0
    for i in range(table.num_rows):
        val = col[i].as_py()
        mask_val = val[0] if isinstance(val, (list, tuple, np.ndarray)) else int(val)
        if mask_val != 0:
            count += 1
    return count


def keep_indices_from_table(table) -> np.ndarray:
    if "step_validity_mask" not in table.column_names:
        return np.arange(table.num_rows, dtype=np.int64)
    col = table.column("step_validity_mask").combine_chunks()
    indices = []
    for i in range(table.num_rows):
        val = col[i].as_py()
        mask_val = val[0] if isinstance(val, (list, tuple, np.ndarray)) else int(val)
        if mask_val != 0:
            indices.append(i)
    return np.asarray(indices, dtype=np.int64)


def filter_table_by_indices(table, indices: np.ndarray):
    if indices.size == 0:
        return table.slice(0, 0)
    return table.take(pa.array(indices, type=pa.int64()))


def write_episode_with_validity_mask(path: Path, table, step_validity_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "step_validity_mask" in table.column_names:
        table = table.drop(["step_validity_mask"])
    mask_col = pa.array(
        [np.array([int(v)], dtype=np.int8) for v in step_validity_mask],
        type=pa.list_(pa.int8(), 1),
    )
    table = table.append_column("step_validity_mask", mask_col)
    pq.write_table(table, path, compression="snappy")
