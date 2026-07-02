# YAML 配置参数说明

本文档说明 pipeline 配置文件中各字段的含义。配置文件采用分层结构，Humanoid 与 EgoDex 共用同一 schema，部分默认值因 embodiment 不同而异。

---

## 配置文件一览

| 文件 | Embodiment | 说明 |
|------|------------|------|
| `config/humanoid_merged.yaml` | humanoid | Agilex CobotMagic2 双臂数据 |
| `config/egodex_add_remove_lid.yaml` | egodex | EgoDex add_remove_lid 子集 |

---

## schema — 数据模式

定义如何从原始 Parquet 读取并转换为 canonical 14 维。

| 参数 | 类型 | 说明 |
|------|------|------|
| `embodiment` | string | 机器人类型标识：`humanoid` / `egodex` |
| `layout` | string | Canonical 语义布局：`joint_gripper`（关节+夹爪）或 `pose_gripper`（xyz+rpy+夹爪） |
| `canonical_dim` | int | 统一维度，固定为 **14** |
| `state_column` | string | Parquet 中 state 列名，默认 `observation.state` |
| `action_column` | string | Parquet 中 action 列名，默认 `action` |

**layout 与 embodiment 对应关系：**

- `humanoid` → `joint_gripper`：state/action 各 14 维 = 12 关节 + 2 夹爪
- `egodex` → `pose_gripper`：20 维 raw（rot6d）→ 14 维 pose

---

## dataset — 数据集路径

| 参数 | 类型 | 说明 |
|------|------|------|
| `root` | string | LeRobot 数据集根目录（含 `data/`、`videos/`、`meta/`） |
| `embodiment` | string | 与 schema.embodiment 一致，用于报告元数据 |
| `robot_type` | string | 机器型号描述，如 `agilex_cobotmagic2`、`egodex` |
| `fps` | int | 帧率，通常 30 |
| `total_episodes` | int | 数据集中 episode 总数，用于枚举索引 |

可被 CLI `--dataset-root` 覆盖。

---

## pipeline — 全局运行参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `num_workers` | int | 128 | 多进程 worker 数量（统计与逐 episode 处理） |
| `output_mode` | string | report | 输出模式：`report`（仅报告）、`filter`（写 parquet）、`both` |
| `min_episode_length` | int | 30 | 最短有效 episode 长度参考（与 stage3 配合） |
| `action_zero_epsilon` | float | 1e-4 | 判定 action 维度「离开零位」的阈值 |
| `stage1_post_zero_grace_frames` | int | 0 | 每个关节 action 刚离开零位后，额外豁免 Stage1 检测的帧数 |
| `discard_short_prefix` | bool | false | 为 true 时，保留帧数 < min_episode_length 则标记 discard |

---

## preprocess — 预处理

| 参数 | 类型 | 说明 |
|------|------|------|
| `action_from_state` | bool | 为 true 时，读取后令 action 等于 state（丢弃原始 action）。Humanoid 通常设为 true；EgoDex 通常 false（state≈action 已成立） |

可被 CLI `--action-from-state` / `--no-action-from-state` 覆盖。

---

## stage1 — 突变检测

### stage1.stats

| 参数 | 类型 | 说明 |
|------|------|------|
| `recompute` | bool | 是否重新计算全局 Stage1 阈值（否则读 cache） |
| `cache_path` | string | 相对 output_dir 的缓存路径，如 `cache/stage1_global_stats.npz` |

### stage1.smoothing

平滑参数，用于 residual / accel / jerk 计算。

| 参数 | 类型 | 说明 |
|------|------|------|
| `median_kernel` | int | 中值滤波窗口（奇数） |
| `savgol_window` | int | Savitzky-Golay 窗口长度（奇数，须 > polyorder） |
| `savgol_polyorder` | int | SG 多项式阶数 |

### stage1.thresholds

| 参数 | 类型 | 说明 |
|------|------|------|
| `k_residual` | float | residual 通道 MAD 缩放系数（hybrid 阈值） |
| `k_accel` | float | 加速度通道 MAD 缩放系数 |
| `k_jerk` | float | jerk 通道 MAD 缩放系数 |
| `percentile_floor` | float | 分位数下限（如 99.999），与 MAD 阈值取 max |

突变判定：residual 超阈 **且**（accel 或 jerk 超阈）。

### stage1.hard_limits

物理硬限幅，超界帧标记为 abnormal。

| 参数 | 类型 | 说明 |
|------|------|------|
| `joint_abs_max` | float | 关节角绝对值上限（joint_gripper layout，前 12 维） |
| `ee_position_max` | float | xyz 绝对值上限（pose_gripper layout） |
| `rpy_abs_max` | float | 欧拉角绝对值上限（弧度，通常 π） |
| `gripper_max` | float | 夹爪开合绝对值上限 |

### stage1.exclusion

| 参数 | 路径 | 说明 |
|------|------|------|
| `mode` | exclusion.mode | 排除策略标识，当前为 `validity_mask` |
| `max_cluster_length` | frame_removal | 帧级：短于该长度的突变簇可过滤（配合 min_cluster） |
| `min_cluster_length_for_abnormal` | frame_removal | 至少多长才计为有效突变簇 |
| `min_cluster_length` | episode_discard | episode 级：连续突变簇超过该长度且跳变足够大 → 整集丢弃 |
| `min_cluster_frame_jump` | episode_discard | 簇内最小帧间跳变幅度 |
| `on_hard_limit_violation` | episode_discard | 硬限幅触发时是否整集 discard |

---

## stage2 — State-Action 趋势对齐

| 参数 | 路径 | 说明 |
|------|------|------|
| `action_type` | stage2 | `absolute`（绝对量）或 `delta`（差分积分后对齐） |
| `state_key` | shared_dims | 逻辑 state 键名（文档用途） |
| `action_key` | shared_dims | 逻辑 action 键名 |
| `num_dims` | shared_dims | 对齐维度数，当前 **14** |
| `median_kernel` | smoothing | 同 stage1 |
| `savgol_window` | smoothing | 同 stage1 |
| `savgol_polyorder` | smoothing | 同 stage1 |
| `max_lag_frames` | alignment | 互相关搜索的最大时滞（帧） |
| `diff_epsilon` | alignment | 差分「有效运动」的最小幅度 |
| `min_active_samples` | alignment | 参与 DA 计算的最少有效样本数 |
| `da_per_dim` | thresholds | 单维 DA 下限，低于则 discard |
| `da_episode_mean` | thresholds | episode 平均 DA 下限 |
| `mode` | exclusion | 固定为 `episode_discard`：不达标整集丢弃 |

**DA (Direction Agreement)**：state 与 action 差分方向一致的比例。

---

## stage3 — 极值 / 分位数带

### stage3.stats

| 参数 | 类型 | 说明 |
|------|------|------|
| `recompute` | bool | 是否重算全局 q01/q99 |
| `cache_path` | string | 如 `cache/global_stats.npz` |
| `num_histogram_bins` | int | 直方图 bin 数（默认 65536） |

### stage3 主体

| 参数 | 类型 | 说明 |
|------|------|------|
| `alpha` | float | 分位数带扩展系数：`[q01 - α·I, q99 + α·I]`，I = q99 - q01 |

### stage3.exempt_dims

以下维度 **不参与** percentile 带检测（仍受硬限幅约束）：

| 参数 | 说明 |
|------|------|
| `gripper_state` | state 中夹爪维度索引 |
| `gripper_action` | action 中夹爪维度索引 |
| `rpy_state` | state 中 rpy 维度索引（EgoDex: 3,4,5,10,11,12） |
| `rpy_action` | action 中 rpy 维度索引 |

Humanoid joint layout 通常 `rpy_*` 为空列表。

### stage3.hard_limits

| 参数 | 类型 | 说明 |
|------|------|------|
| `joint` | [lo, hi] | 关节角范围（joint_gripper） |
| `ee_xyz` | [lo, hi] | 末端 xyz 范围（pose_gripper）；**EgoDex 须含负值**，如 `[-2, 2]` |
| `rpy` | [lo, hi] | 欧拉角范围（弧度） |
| `gripper` | [lo, hi] | 夹爪范围 |

### stage3.quaternion

历史兼容字段；当前 canonical pipeline 对四元数直接使用 rpy，**Humanoid joint 模式下不生效**。保留以备扩展。

| 参数 | 说明 |
|------|------|
| `norm_min` / `norm_max` | 四元数模长范围 |
| `component_abs_max` | 四元数分量绝对值上限 |

### stage3.exclusion

| 参数 | 说明 |
|------|------|
| `mode` | `frame_removal`：异常帧参与前缀截断 |
| `min_episode_length` | 去掉异常帧后剩余帧数低于此值则 discard |
| `validity_mask` | `prefix_truncate`：从首异常帧起截断 |

---

## stage4 — 静止区间缩短

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | bool | true | 是否启用 Stage4 |
| `max_static_steps` | int | 5 | 每段连续静止最多保留的帧数 |
| `change_epsilon` | float | 0.0 | state/action 相等判定容差；0 表示精确相等 |

**静止定义**：某帧 state 与 action 逐元素相同（在 epsilon 内）。静止段超过 `max_static_steps` 的部分会被删除（稀疏 mask）。

---

## stage5 — 坐标对齐

以 **第一帧 camera_top** 为原点，输出统一 14 维 pose_gripper。

| 参数 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 是否启用；EgoDex 示例配置默认 false，Humanoid 默认 true |
| `reference_frame` | string | 参考系名称，固定语义 `camera_top_frame0` |
| `rotation_correction_euler_xyz` | [r,p,y] | 额外旋转校正（弧度，extrinsic xyz），使 **+X 对齐机器人前方** |
| `egodex_extrinsics_column` | string | EgoDex 逐帧相机外参列，16 维 = 4×4 行优先 |
| `humanoid_calibration_relpath` | string | Humanoid 标定文件相对 dataset.root 的路径模板 |
| `humanoid_camera_extrinsic_key` | string | 标定 JSON 中外参键，默认 `camera_front_to_arm_left`（top 相机） |

**rotation_correction 示例：**

```yaml
rotation_correction_euler_xyz: [0.0, -1.5707963267948966, 0.0]  # 绕 Y 轴 -90°
```

**输出：** `data_aligned/chunk-XXX/episode_XXXXXX.npz`（需 `output_mode: filter/both`）

---

## CLI 与 YAML 的对应关系

| CLI 参数 | 覆盖的 YAML 字段 |
|----------|------------------|
| `--dataset-root` | `dataset.root` |
| `--output-dir` | （写入 `PipelineConfig.output_dir`） |
| `--output-mode` | `pipeline.output_mode` |
| `--num-workers` | `pipeline.num_workers` |
| `--recompute-stats` | `stage1.stats.recompute` + `stage3.stats.recompute` |
| `--stats-cache` | `stage3.stats.cache_path`（相对路径需配合 output_dir） |
| `--action-from-state` | `preprocess.action_from_state` |

未在 CLI 暴露的参数仅能通过修改 YAML 调整。

---

## Humanoid vs EgoDex 推荐差异

| 配置项 | Humanoid | EgoDex |
|--------|----------|--------|
| `schema.layout` | joint_gripper | pose_gripper |
| `preprocess.action_from_state` | true | false |
| `pipeline.min_episode_length` | 30 | 15 |
| `stage1_post_zero_grace_frames` | 60 | 30 |
| `stage3.exempt_dims.gripper_*` | [12, 13] | [6, 13] |
| `stage3.exempt_dims.rpy_*` | [] | [3,4,5,10,11,12] |
| `stage3.hard_limits.ee_xyz` | [-2, 2] | [-2, 2] |
| `stage5.enabled` | true（按需） | false（按需） |
| `stage5` 外参来源 | calibration bundle | camera_extrinsics_world |

---

## 缓存文件格式

### cache/global_stats.npz

Stage3 全局统计：`state_q01/q99`、`action_q01/q99`、min/max、帧数。

### cache/stage1_global_stats.npz

Stage1 全局阈值：`thr_residual`、`thr_accel`、`thr_jerk`（各 28 维）、`num_channels`。

删除或 `--recompute-stats` 可强制重算。
