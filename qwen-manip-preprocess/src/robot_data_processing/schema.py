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

# Legacy constants
STATE_DIM = CANONICAL_DIM
ACTION_DIM = CANONICAL_DIM
ARM_DIM = 12
HUMANOID_STATE_DIM = 28
HUMANOID_ACTION_DIM = 14
HUMANOID_ALIGNMENT_DIM = 14
ROBOMIND_UR_STATE_DIM = 26
ROBOMIND_UR_ACTION_DIM = 52
ROBOMIND_UR_ALIGNMENT_DIM = 14
ROBOMIND_UR_JOINT_INDICES = tuple(range(14, 26))
ROBOMIND_UR_GRIPPER_INDICES = (0, 7)

ROBOMIND_UR_RAW_COLUMNS = (
    "puppet/arm_left_position_align",
    "puppet/arm_right_position_align",
    "puppet/end_effector_left_position_align",
    "puppet/end_effector_right_position_align",
    "master/arm_left_position_align",
    "master/arm_right_position_align",
    "master/end_effector_left_position_align",
    "master/end_effector_right_position_align",
)

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
    """Embodiment-specific schema for pipeline state/action dimensions."""

    embodiment: str
    layout: str  # "joint_gripper" | "pose_gripper"
    canonical_dim: int = CANONICAL_DIM
    state_dim: int | None = None
    action_dim: int | None = None
    alignment_dims: int | None = None
    joint_index_list: tuple[int, ...] | None = None
    gripper_index_list: tuple[int, ...] | None = None
    action_joint_index_list: tuple[int, ...] | None = None
    action_gripper_index_list: tuple[int, ...] | None = None
    raw_columns: tuple[str, ...] = HUMANOID_RAW_COLUMNS
    state_column: str = "observation.state"
    action_column: str = "action"

    @property
    def pipeline_state_dim(self) -> int:
        if self.state_dim is not None:
            return self.state_dim
        if self.embodiment == "humanoid":
            return HUMANOID_STATE_DIM
        if self.embodiment == "robomind_ur":
            return ROBOMIND_UR_STATE_DIM
        return self.canonical_dim

    @property
    def pipeline_action_dim(self) -> int:
        if self.action_dim is not None:
            return self.action_dim
        if self.embodiment == "humanoid":
            return HUMANOID_ACTION_DIM
        if self.embodiment == "robomind_ur":
            return ROBOMIND_UR_ACTION_DIM
        return self.canonical_dim

    @property
    def alignment_dim(self) -> int:
        if self.alignment_dims is not None:
            return self.alignment_dims
        if self.embodiment == "humanoid":
            return HUMANOID_ALIGNMENT_DIM
        if self.embodiment == "robomind_ur":
            return ROBOMIND_UR_ALIGNMENT_DIM
        return self.canonical_dim

    @property
    def stage1_num_channels(self) -> int:
        return self.pipeline_state_dim + self.pipeline_action_dim

    @property
    def xyz_indices(self) -> tuple[int, ...]:
        return XYZ_INDICES if self.layout == "pose_gripper" else ()

    @property
    def rpy_indices(self) -> tuple[int, ...]:
        return RPY_INDICES if self.layout == "pose_gripper" else ()

    @property
    def joint_indices(self) -> tuple[int, ...]:
        if self.joint_index_list is not None:
            return self.joint_index_list
        return JOINT_INDICES if self.layout == "joint_gripper" else ()

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        if self.gripper_index_list is not None:
            return self.gripper_index_list
        return GRIPPER_INDICES if self.layout == "pose_gripper" else (12, 13)

    @property
    def action_joint_indices(self) -> tuple[int, ...]:
        if self.action_joint_index_list is not None:
            return self.action_joint_index_list
        return self.joint_indices

    @property
    def action_gripper_indices(self) -> tuple[int, ...]:
        if self.action_gripper_index_list is not None:
            return self.action_gripper_index_list
        return self.gripper_indices

    @property
    def stage1_state_channel_offset(self) -> int:
        return 0

    @property
    def stage1_action_channel_offset(self) -> int:
        return self.pipeline_state_dim


HUMANOID_SCHEMA = DatasetSchema(
    embodiment="humanoid",
    layout="joint_gripper",
    state_dim=HUMANOID_STATE_DIM,
    action_dim=HUMANOID_ACTION_DIM,
    alignment_dims=HUMANOID_ALIGNMENT_DIM,
    raw_columns=HUMANOID_RAW_COLUMNS,
)

EGODEX_SCHEMA = DatasetSchema(
    embodiment="egodex",
    layout="pose_gripper",
    raw_columns=EGODEX_RAW_COLUMNS,
)

ROBOMIND_UR_SCHEMA = DatasetSchema(
    embodiment="robomind_ur",
    layout="joint_gripper",
    state_dim=ROBOMIND_UR_STATE_DIM,
    action_dim=ROBOMIND_UR_ACTION_DIM,
    alignment_dims=ROBOMIND_UR_ALIGNMENT_DIM,
    joint_index_list=ROBOMIND_UR_JOINT_INDICES,
    gripper_index_list=ROBOMIND_UR_GRIPPER_INDICES,
    action_joint_index_list=ROBOMIND_UR_JOINT_INDICES,
    action_gripper_index_list=ROBOMIND_UR_GRIPPER_INDICES,
    raw_columns=ROBOMIND_UR_RAW_COLUMNS,
)

SCHEMA_REGISTRY: dict[str, DatasetSchema] = {
    "humanoid": HUMANOID_SCHEMA,
    "egodex": EGODEX_SCHEMA,
    "robomind_ur": ROBOMIND_UR_SCHEMA,
}


def schema_from_yaml(yaml_cfg: dict[str, Any]) -> DatasetSchema:
    ds = yaml_cfg.get("schema", {})
    embodiment = ds.get("embodiment", yaml_cfg.get("dataset", {}).get("embodiment", "humanoid"))
    if embodiment in SCHEMA_REGISTRY and not ds:
        return SCHEMA_REGISTRY[embodiment]
    layout = ds.get("layout", "joint_gripper" if embodiment == "humanoid" else "pose_gripper")
    raw_columns = tuple(
        ds.get(
            "raw_columns",
            EGODEX_RAW_COLUMNS
            if embodiment == "egodex"
            else ROBOMIND_UR_RAW_COLUMNS
            if embodiment == "robomind_ur"
            else HUMANOID_RAW_COLUMNS,
        )
    )

    def _idx(key: str) -> tuple[int, ...] | None:
        val = ds.get(key)
        return tuple(val) if val is not None else None

    return DatasetSchema(
        embodiment=embodiment,
        layout=layout,
        canonical_dim=int(ds.get("canonical_dim", CANONICAL_DIM)),
        state_dim=int(ds["state_dim"]) if ds.get("state_dim") is not None else None,
        action_dim=int(ds["action_dim"]) if ds.get("action_dim") is not None else None,
        alignment_dims=int(ds["alignment_dims"]) if ds.get("alignment_dims") is not None else None,
        joint_index_list=_idx("joint_indices"),
        gripper_index_list=_idx("gripper_indices"),
        action_joint_index_list=_idx("action_joint_indices"),
        action_gripper_index_list=_idx("action_gripper_indices"),
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
