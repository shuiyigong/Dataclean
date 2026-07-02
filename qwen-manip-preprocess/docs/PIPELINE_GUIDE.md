# Robot Data Processing Pipeline 使用说明

本工程对 LeRobot v2.1 格式的机器人数据进行质量检测、帧过滤、坐标对齐与导出。支持 **Humanoid**（Agilex CobotMagic2）与 **EgoDex** 两种 embodiment，通过 YAML 配置驱动。

---

## 目录结构

```
robot_data_processing/
├── config/                    # 数据集配置文件
│   ├── humanoid_merged.yaml
│   └── egodex_add_remove_lid.yaml
├── docs/                      # 文档
│   ├── PIPELINE_GUIDE.md      # 本文档
│   └── CONFIG_REFERENCE.md    # YAML 参数说明
├── scripts/                   # 命令行入口
│   ├── run_pipeline.py        # 主 pipeline
│   ├── export_lerobot.py      # LeRobot 完整导出
│   ├── analyze_experiment.py  # 实验结果分析
│   ├── analyze_stage1_metrics.py
│   └── preprocess_action_from_state.py
├── src/robot_data_processing/ # 核心库
└── output/                    # 运行输出（示例）
```

---

## 环境准备

```bash
cd /path/to/robot_data_processing
pip install -r requirements.txt   # 或 pip install -e .
export PYTHONPATH=src
```

依赖主要包括：`numpy`、`pyarrow`、`scipy`、`pyyaml`、`tqdm`。

---

## 数据格式要求

输入数据集需为 **LeRobot v2.1** 布局：

```
dataset_root/
├── data/chunk-XXX/episode_XXXXXX.parquet
├── videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4
├── meta/info.json
└── parameters/...              # Humanoid 可选，含相机标定
```

### Canonical 14 维表示

Pipeline 内部将 state / action 统一为 **14 维**：

| 索引 | 含义 |
|------|------|
| 0–2  | 左手 xyz |
| 3–5  | 左手 rpy（欧拉角，弧度） |
| 6    | 左夹爪 |
| 7–9  | 右手 xyz |
| 10–12| 右手 rpy |
| 13   | 右夹爪 |

**Embodiment 转换规则：**

| Embodiment | 原始 state | 转换 |
|------------|-----------|------|
| Humanoid   | 28 维（关节+EE） | state/action 取前 14 维（12 关节 + 2 夹爪） |
| EgoDex     | 20 维（xyz+rot6d+width×2） | rot6d → 欧拉角，得到 14 维 pose |

详细 schema 配置见 `CONFIG_REFERENCE.md`。

---

## Pipeline 流程概览

```
读取 Parquet → 预处理(action=state) → Canonical 转换
    ↓
Stage 1  突变检测（全局 per-channel 阈值）
    ↓
Stage 2  State-Action 趋势对齐（DA）
    ↓
Stage 3  极值 / 分位数带检测
    ↓
Stage 4  静止区间缩短
    ↓
Stage 5  坐标对齐（camera_top 第 0 帧为原点）  [可选]
    ↓
输出 report / filtered parquet / aligned npz
```

### 有效性掩码（step_validity_mask）

- **Stage 1–3 异常**：从首个异常帧起 **前缀截断**（该帧及之后标记为 0）
- **Stage 4**：在保留前缀内，对连续静止超过阈值的区间 **删帧**（稀疏删除）
- 最终 mask：`1` = 保留，`0` = 丢弃

---

## 各 Stage 功能说明

### 预处理（preprocess）

- `action_from_state: true` 时，丢弃原始 action，令 `action = state`（Humanoid 为 arm+gripper 子列）
- 在读取 Parquet 时生效，不影响原始磁盘文件（除非单独运行 preprocess 脚本）

### Stage 1：突变检测

- 对 canonical state（14 维）+ action（14 维）共 **28 个通道** 计算 residual / accel / jerk
- 使用 **全局统计**（两遍 histogram）生成 hybrid 阈值（MAD + 分位数下限）
- 检测 joint/xyz 硬限幅、长连续突变簇
- 支持 per-dim 启动段排除（action 接近零的帧）

### Stage 2：趋势对齐

- 对 state 与 action 各维做平滑后，计算 **Direction Agreement (DA)**
- 允许 ±`max_lag_frames` 帧时滞
- DA 低于阈值则 **整集丢弃**

### Stage 3：极值检测

- 基于全局 q01/q99 的扩展分位数带（`alpha`）
- 硬限幅：关节 / xyz / rpy / 夹爪范围
- 超出范围的帧标记为异常（参与前缀截断）
- 夹爪、rpy 等维度可配置豁免

### Stage 4：静止区间缩短

- 检测 state 与 action **完全相同**的连续静止段
- 每段最多保留 `max_static_steps` 帧，其余删除
- 不改变前缀截断逻辑，仅在前缀有效范围内稀疏删帧

### Stage 5：坐标对齐

- 以 **第一帧 camera_top** 为坐标原点与参考朝向
- 输出统一 **14 维 pose_gripper**（xyz + rpy + gripper）
- 应用 per-dataset 旋转校正，使 **+X 对齐机器人前方**
- **EgoDex**：使用 `observation.camera_extrinsics_world`
- **Humanoid**：使用 `parameters/.../calibration_bundle_optimized.json` 中 top 相机外参 + EE 位姿
- 启用且 `output_mode` 为 `filter`/`both` 时，写入 `data_aligned/*.npz`

---

## 主命令：run_pipeline

```bash
cd /path/to/robot_data_processing
PYTHONPATH=src python scripts/run_pipeline.py \
  --config config/humanoid_merged.yaml \
  --output-dir output/my_experiment \
  [选项...]
```

### 常用 CLI 参数

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径 |
| `--output-dir` | **必填**，输出目录 |
| `--dataset-root` | 覆盖 YAML 中的 dataset.root |
| `--output-mode` | `report` / `filter` / `both` |
| `--num-workers` | 并行 worker 数 |
| `--episode-limit N` | 只处理前 N 条 episode |
| `--episode-indices` | 逗号分隔的 episode 索引，如 `0,1,5` |
| `--stats-episodes` | `all` 或 `processed`；统计集范围 |
| `--recompute-stats` | 强制重算 Stage1/Stage3 全局统计 |
| `--no-recompute-stats` | 使用已有 cache |
| `--action-from-state` | 启用 action=state 预处理 |
| `--no-action-from-state` | 禁用（即使 YAML 中开启） |
| `--quiet` | 关闭 tqdm 进度条 |

### 示例

**Humanoid 500 条实验：**

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config config/humanoid_merged.yaml \
  --output-dir output/exp500 \
  --episode-limit 500 \
  --recompute-stats \
  --output-mode both
```

**EgoDex 1000 条：**

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config config/egodex_add_remove_lid.yaml \
  --output-dir output/egodex_1000 \
  --episode-limit 1000 \
  --recompute-stats
```

**子集处理、统计仍用全量：**

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config config/humanoid_merged.yaml \
  --output-dir output/subset_test \
  --episode-limit 100 \
  --stats-episodes all
```

---

## 输出说明

运行后在 `--output-dir` 下生成：

```
output/my_experiment/
├── cache/
│   ├── global_stats.npz           # Stage3 全局分位数
│   └── stage1_global_stats.npz    # Stage1 全局阈值
├── reports/
│   ├── quality_report.json        # 汇总报告
│   └── exclusion_log.jsonl        # 逐 episode 明细
├── data_filtered/                 # output_mode: filter/both
│   └── chunk-XXX/episode_XXXXXX.parquet  # 带 step_validity_mask
└── data_aligned/                  # stage5.enabled + filter/both
    └── chunk-XXX/episode_XXXXXX.npz
```

### quality_report.json 主要字段

- `summary.total_episodes` / `kept_frames` / `valid_prefix_frame_rate`
- `summary.stage1_flagged_frames` / `stage3_excluded_frames` / `stage4_removed_frames`
- `stage2_da`：趋势对齐得分分布

### data_aligned npz 内容

- `aligned_state` / `aligned_action`：`(T, 14)` float32，camera_top 第 0 帧坐标系
- `step_validity_mask`：与 filtered 一致
- `reference_frame` / `rotation_correction_euler_xyz`

---

## LeRobot 导出：export_lerobot

将 pipeline 过滤结果导出为完整 LeRobot 数据集（同步裁剪 parquet、video、meta）：

```bash
PYTHONPATH=src python scripts/export_lerobot.py \
  --source-root /path/to/original/dataset \
  --filtered-root output/my_experiment \
  --output-root output/my_experiment_lerobot \
  --episode-limit 100
```

| 参数 | 说明 |
|------|------|
| `--source-root` | 原始 LeRobot 数据集根目录 |
| `--filtered-root` | pipeline 输出目录（含 `data_filtered/`） |
| `--output-root` | 导出目标目录 |
| `--episode-limit` / `--episode-indices` | 导出范围 |
| `--skip-video-stats` | 跳过 ffmpeg 视频统计 |
| `--verify-only` | 仅做对齐验证 |

导出完成后生成 `alignment_report.json`，验证帧数与 video 对齐。

---

## 实验分析脚本

### analyze_experiment.py

对 pipeline 输出做 Stage1 全局阈值分析与可视化：

```bash
PYTHONPATH=src python scripts/analyze_experiment.py \
  --exp-dir output/my_experiment \
  --episode-limit 500
```

输出至 `output/my_experiment/analysis/`（含 JSON 报告与 PNG 图表）。

### preprocess_action_from_state.py

将 `action=state` 预处理 **物化写入** 新 Parquet（不经过 quality pipeline）：

```bash
PYTHONPATH=src python scripts/preprocess_action_from_state.py \
  --source-root /path/to/dataset \
  --output-root /path/to/preprocessed \
  --episode-limit 1000 \
  --num-workers 32
```

---

## 新增 Embodiment 步骤

1. 在 `schema.py` 注册 `DatasetSchema`（layout、raw_columns）
2. 在 `transforms.py` 实现 raw → canonical 14 维转换
3. 复制 YAML 模板，调整 `schema`、`stage3.exempt_dims`、硬限幅
4. 若需 Stage5，提供相机外参来源（逐帧列或标定文件）
5. 小批量试跑 + 检查 `quality_report.json`

---

## 常见问题

**Q: Stage3 删帧过多？**  
检查 `hard_limits.ee_xyz` 是否过窄（EgoDex xyz 可为负，应使用 `[-2, 2]`）。

**Q: 子集实验统计不稳定？**  
小批量试跑时默认在 processed 子集上算统计；全量生产建议 `--stats-episodes all` 或不设 `--episode-limit`。

**Q: Humanoid Stage5 精度？**  
依赖静态标定 bundle 与 EE 位姿同系假设，无逐帧 head pose；详见 Stage5 配置说明。

**Q: cache 如何失效？**  
使用 `--recompute-stats`，或删除 `output/.../cache/*.npz`。

---

## 相关文档

- [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md) — YAML 各字段含义
- 配置文件：`config/humanoid_merged.yaml`、`config/egodex_add_remove_lid.yaml`
