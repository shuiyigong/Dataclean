from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_data_processing.transforms import (
    CANONICAL_DIM,
    GRIPPER_INDICES,
    JOINT_INDICES,
    RPY_INDICES,
    XYZ_INDICES,
)

# Legacy constants (canonical pipeline uses CANONICAL_DIM for both state and action)
STATE_DIM = CANONICAL_DIM
ACTION_DIM = CANONICAL_DIM
ARM_DIM = 12

HUMANOID_RAW_COLUMNS = (
    "observation.state",
    "observation.state.arm.position",
    "observation.state.effector.position",
    "observation.state.end.position",
    "action",
    "action.arm.position",
    "action.effector.position",
)

EGODEX_RAW_COLUMNS = (
    "observation.state",
    "action",
)

PARQUET_FILTER_COLUMNS = (
    "observation.state",
    "observation.state.arm.position",
    "observation.state.effector.position",
    "observation.state.end.position",
    "action",
    "action.arm.position",
    "action.effector.position",
    "observation.confidence",
    "observation.camera_extrinsics_world",
    "observation.camera_intrinsics",
    "timestamps",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


@dataclass(frozen=True)
class DatasetSchema:
    """Embodiment-specific schema driving canonical 14-dim state/action."""

    embodiment: str
    layout: str  # "joint_gripper" | "pose_gripper"
    canonical_dim: int = CANONICAL_DIM
    raw_columns: tuple[str, ...] = HUMANOID_RAW_COLUMNS
    state_column: str = "observation.state"
    action_column: str = "action"

    @property
    def stage1_num_channels(self) -> int:
        return 2 * self.canonical_dim

    @property
    def xyz_indices(self) -> tuple[int, ...]:
        return XYZ_INDICES if self.layout == "pose_gripper" else ()

    @property
    def rpy_indices(self) -> tuple[int, ...]:
        return RPY_INDICES if self.layout == "pose_gripper" else ()

    @property
    def joint_indices(self) -> tuple[int, ...]:
        return JOINT_INDICES if self.layout == "joint_gripper" else ()

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        return GRIPPER_INDICES if self.layout == "pose_gripper" else (12, 13)

    @property
    def stage1_state_channel_offset(self) -> int:
        return 0

    @property
    def stage1_action_channel_offset(self) -> int:
        return self.canonical_dim


HUMANOID_SCHEMA = DatasetSchema(
    embodiment="humanoid",
    layout="joint_gripper",
    raw_columns=HUMANOID_RAW_COLUMNS,
)

EGODEX_SCHEMA = DatasetSchema(
    embodiment="egodex",
    layout="pose_gripper",
    raw_columns=EGODEX_RAW_COLUMNS,
)

SCHEMA_REGISTRY: dict[str, DatasetSchema] = {
    "humanoid": HUMANOID_SCHEMA,
    "egodex": EGODEX_SCHEMA,
}


def schema_from_yaml(yaml_cfg: dict[str, Any]) -> DatasetSchema:
    ds = yaml_cfg.get("schema", {})
    embodiment = ds.get("embodiment", yaml_cfg.get("dataset", {}).get("embodiment", "humanoid"))
    if embodiment in SCHEMA_REGISTRY and not ds:
        return SCHEMA_REGISTRY[embodiment]
    layout = ds.get("layout", "joint_gripper" if embodiment == "humanoid" else "pose_gripper")
    raw_columns = tuple(ds.get("raw_columns", EGODEX_RAW_COLUMNS if embodiment == "egodex" else HUMANOID_RAW_COLUMNS))
    return DatasetSchema(
        embodiment=embodiment,
        layout=layout,
        canonical_dim=int(ds.get("canonical_dim", CANONICAL_DIM)),
        raw_columns=raw_columns,
        state_column=ds.get("state_column", "observation.state"),
        action_column=ds.get("action_column", "action"),
    )


def stack_column(values: np.ndarray | list) -> np.ndarray:
    """Stack parquet object column to (T, D) float64 array."""
    if isinstance(values, np.ndarray) and values.dtype != object:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr
    return np.stack(values).astype(np.float64, copy=False)


def max_contiguous_run(flags: np.ndarray) -> int:
    """Return length of longest contiguous True run."""
    if flags.size == 0 or not flags.any():
        return 0
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int((ends - starts).max())


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-12
