from __future__ import annotations

import pyarrow as pa

import numpy as np

from robot_data_processing.schema import ACTION_DIM, DatasetSchema, HUMANOID_SCHEMA, stack_column


def replace_action_with_state(
    data: dict[str, np.ndarray],
    schema: DatasetSchema = HUMANOID_SCHEMA,
) -> dict[str, np.ndarray]:
    """Discard original action; set action equal to state (canonical or raw columns)."""
    if schema.embodiment == "humanoid":
        state = data["observation.state"]
        if "observation.state.arm.position" in data:
            data["action.arm.position"] = data["observation.state.arm.position"].copy()
        if "observation.state.effector.position" in data:
            data["action.effector.position"] = data["observation.state.effector.position"].copy()
        if state.shape[1] >= ACTION_DIM:
            data["action"] = state[:, :ACTION_DIM].copy()
        elif "observation.state.arm.position" in data and "observation.state.effector.position" in data:
            data["action"] = np.concatenate(
                [data["observation.state.arm.position"], data["observation.state.effector.position"]],
                axis=1,
            )
        return data

    state = data[schema.state_column]
    data[schema.action_column] = state.copy()
    return data


def _list_array_from_2d(arr: np.ndarray) -> pa.Array:
    return pa.array([row.astype(np.float32).tolist() for row in arr])


def replace_action_with_state_table(
    table: pa.Table,
    schema: DatasetSchema = HUMANOID_SCHEMA,
) -> pa.Table:
    """Replace action columns in a parquet table with state values."""
    data: dict[str, np.ndarray] = {}
    if schema.embodiment == "humanoid":
        col_names = (
            "observation.state",
            "observation.state.arm.position",
            "observation.state.effector.position",
            "action",
            "action.arm.position",
            "action.effector.position",
        )
    else:
        col_names = (schema.state_column, schema.action_column)

    for name in col_names:
        if name not in table.column_names:
            continue
        col = table.column(name).combine_chunks()
        data[name] = stack_column(col.to_numpy(zero_copy_only=False))

    replace_action_with_state(data, schema)

    out = table
    if schema.embodiment == "humanoid":
        for name in ("action.arm.position", "action.effector.position", "action"):
            if name not in out.column_names:
                continue
            idx = out.column_names.index(name)
            out = out.set_column(idx, name, _list_array_from_2d(data[name]))
    else:
        idx = out.column_names.index(schema.action_column)
        out = out.set_column(idx, schema.action_column, _list_array_from_2d(data[schema.action_column]))
    return out
