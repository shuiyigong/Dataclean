# EgoDex → 机器人动作数据处理管线规范（供 Codex 实现）

## 0. 背景与目标

EgoDex是一个用 Apple Vision Pro 采集的大规模第一视角人手操作数据集：
- 829 小时视频，30Hz，1080p
- 338,000 episodes，194 个任务
- 标注：每帧的 ARKit 骨架，包括上半身 20 个关节 + 每只手 25 个关节，均为 4×4 的 SE(3) 齐次变换矩阵
- 每个关节还有一个 0~1 的 confidence 值（wrist 的 confidence 表示整只手是否被检测到；手指关节的 confidence 是相对 wrist 的）

**目标**：把 EgoDex 的人手关节轨迹，转换成可以用于训练平行夹爪机器人（parallel-jaw gripper）策略的 state-action 序列，并完成跨数据集风格的质量过滤、归一化，最终产出统一格式的训练集。

整个管线分三大块：
1. **关键点提取**（从 ARKit 骨架取出需要的点）
2. **Action Alignment**（人手 → 夹爪的轨迹重定向 + 平滑）
3. **五阶段数据过滤与归一化**（跨数据集质量管控，适配多 embodiment 聚合训练场景）


---

## 1. 数据读取层

### 1.1 输入格式
数据集结构大致为：
part1.zip / part2.zip / ... / test.zip / extra.zip
└── <task_name>/
    └── <episode_id>/
        ├── <episode_id>.mp4          # 1920x1080, 30fps RGB视频
        └── <episode_id>.hdf5 (或类似) # 每帧的关节SE(3)、相机内外参、confidence、语言标注
```
/mnt/project_rlinf/runze/egodex_demo有一个小样本，先对这个小样本做处理。hdf5文件中的每一个关节每一帧的transform都是相对于一个并不稳定的世界坐标系的，我需要先把需要用的关节映射到相机系中，后面的处理都在相机系处理。同时保存每个episode camera本身相对世界系得transform。


### 1.2 关节命名
左手关节（右手同名前缀换成 right）：
```
leftIndexFingerMetacarpal, leftIndexFingerKnuckle,
leftIndexFingerIntermediateBase, leftIndexFingerIntermediateTip, leftIndexFingerTip,
leftMiddleFinger... (同上5个), leftRingFinger...(同上5个), leftLittleFinger...(同上5个),
leftThumbKnuckle, leftThumbIntermediateBase, leftThumbIntermediateTip, leftThumbTip
```
上半身：`hip, spine1..spine7, neck1..neck4, leftShoulder, leftArm, leftForearm, leftHand, rightShoulder, rightArm, rightForearm, rightHand`

**重要**：`leftHand` / `rightHand` 在该数据集里指的是**手腕**（wrist），不是手掌中心。

### 1.3 Task 1: 编写 `egodex_loader.py`
功能：
- 给定 episode 路径，返回：
  - `joints: Dict[str, np.ndarray]`，每个关节 `(T, 4, 4)` 的位姿矩阵序列
  - `confidence: Dict[str, np.ndarray]`，每个关节 `(T,)` 的置信度
  - `meta: dict`（task name, language description, fps=30, camera intrinsics/extrinsics `(T, ...)`）

---

## 2. 关键点提取层

### 2.1 需要的关键点
对单只手（以右手为例，左手对称）：
- `k_thumb = rightThumbTip 的位置`（取4x4矩阵的平移部分 `[:3,3]`）
- `k_index = rightIndexFingerTip 的位置`
- `k_middle = rightMiddleFingerTip 的位置`
- `k_wrist = rightHand 的位置`

### 2.2 Task 2: 编写 `extract_keypoints.py`
```python
def extract_hand_keypoints(joints: dict, hand: str) -> dict:
    """
    hand: 'left' or 'right'
    返回 {'thumb': (T,3), 'index': (T,3), 'middle': (T,3), 'wrist': (T,3)}
    """
```
对两只手都跑一遍，得到双手共 8 条轨迹。


---

## 3. Action Alignment（轨迹重定向到夹爪空间）


### 3.1 虚拟手指（公式1）
```python
k_vf = 0.7 * k_index + 0.3 * k_middle      # (T,3)
p_t  = 0.5 * (k_thumb + k_vf)               # 末端执行器位置 (T,3)
w_t  = np.linalg.norm(k_thumb - k_vf, axis=-1)  # 夹爪宽度 (T,)
```

### 3.2 夹爪姿态（公式2）
```python
s = +1 if hand == 'right' else -1
z = s * (k_thumb - k_vf) / w_t[:, None]     # 抓取轴（沿jaw线）
d = k_vf - k_wrist                          # 手腕到指尖方向
y_raw = np.cross(z, d)
y = y_raw / np.linalg.norm(y_raw, axis=-1, keepdims=True)  # 夹爪法向（jaw平面法线）
x = np.cross(y, z)                          # 接近方向，补全右手系
R_t = np.stack([x, y, z], axis=-1)          # (T,3,3) 每帧一个旋转矩阵
```
**注意符号翻转 `s`**：左右手必须映射到同一套夹爪坐标系约定，否则训练出来的策略左右手会镜像错乱。

### 3.3 Task 3: 编写 `action_alignment.py`
```python
def retarget_to_gripper(keypoints: dict, hand: str) -> dict:
    """
    输入: extract_hand_keypoints 的输出
    输出: {'position': (T,3), 'rotation': (T,3,3), 'width': (T,)}
    """
```
**单元测试要点**：
- 检查每帧 `R_t` 是否正交（`R @ R.T ≈ I`），且 `det(R) ≈ +1`
- 检查 `w_t` 全程非负
- 用左手和右手各跑一个对称的合成测试用例，确认 `s` 的符号翻转生效（两手的 z 轴应该指向同一物理方向）

### 3.4 平滑滤波（论文原文要求）
- 位置 `p_t` 和宽度 `w_t`：用 **Savitzky-Golay** 滤波（`scipy.signal.savgol_filter`），窗口长度和多项式阶数留作可调超参数（建议窗口 9~15 帧 @30Hz，2阶多项式，先用可视化检查再定）。
- 旋转 `R_t`：先转四元数，用**高斯加权 SLERP**做平滑——不能直接对四元数分量做线性滤波（会破坏单位长度约束）。实现思路：
```python
def gaussian_weighted_slerp_smooth(quats: np.ndarray, sigma: float) -> np.ndarray:
    """
    对每个时刻t，用以t为中心、标准差sigma的高斯核对邻域四元数做加权球面平均。
    可用 scipy.spatial.transform.Rotation + 迭代Karcher mean 实现，
    或简化为：用高斯核权重做相邻两两SLERP的级联近似。
    """
```
建议直接调用 `scipy.spatial.transform.Rotation.mean(weights=...)`（scipy ≥1.x 支持加权平均旋转，本质是黎曼重心，等价于这里要的平滑）。

### 3.5 Task 4: 编写 `smoothing.py`，对 Task 3 输出做后处理，产出最终 48 维（如果双手）或 9 维（单手：3位置+6D旋转(取R的前两列)+1宽度）动作序列。

> 旋转表示的选择：论文 Section 4.1 用的是 6D 表示（R 的前两列，按 Zhou et al. 2019 连续旋转表示），训练时建议跟随这个约定，而不是用完整 3x3 或四元数，因为 6D 表示在网络回归时数值更稳定。

---

## 4. 五阶段数据过滤与归一化管线（对应你截图 Image 2）

这一节是**多数据集聚合训练**场景下的通用质量管控流程，不是 EgoDex 专属，但如果你要把 EgoDex 和别的机器人数据集（DROID、RoboMIND 等）混合训练，需要套用同一套。

> 如果你只训练 EgoDex 自己的 benchmark（trajectory prediction / inverse dynamics，论文Section 4），**可以跳过 Stage 2 和 Stage 4**（这两个是跨 embodiment 才有意义的检查），但 Stage 1、3、5 仍然有意义（噪声帧、极值、坐标系约定）。

### Stage 1: 突变检测（Sudden Change Detection）
对每个信号维度（position 各轴、width、旋转的某种标量化表示）：
1. 用 cascaded median filter + Savitzky-Golay 得到平滑趋势
2. 计算三个偏差信号：
   - `residual = |raw - smoothed|`
   - `accel = 二阶差分(raw)`
   - `jerk = 三阶差分(raw)`
3. 标记规则：`residual > thresh_r AND (accel > thresh_a OR jerk > thresh_j)`
4. 阈值按"数据来源类型"（real vs sim）、关节类型设置，建议先用全数据集该维度的分布画直方图，取一个高百分位（如 p99.5）作初始阈值，再人工抽查若干被标记帧。

```python
def detect_sudden_changes(raw, smoothed, thresh_r, thresh_a, thresh_j):
    residual = np.abs(raw - smoothed)
    accel = np.diff(raw, n=2, axis=0)
    jerk = np.diff(raw, n=3, axis=0)
    # pad对齐长度后做布尔与/或
    flagged = (residual > thresh_r) & ((accel_padded > thresh_a) | (jerk_padded > thresh_j))
    return flagged
```
EgoDex 场景：这一步主要捕捉手部追踪的瞬时跳变（遮挡导致的关节"跳到错误位置"），可结合 confidence 值——如果某帧 confidence 很低，优先怀疑该帧。

### Stage 2: 状态-动作时序对齐（State-Action Trend Alignment）
对 EgoDex 而言，"state"和"action"的区别不大（都是同一份关节轨迹推出来的），**这一步主要适用于真实机器人数据集**（state来自传感器读数，action来自控制指令，两者可能因时钟不同步而错位）。如果你的下游任务是直接用 EgoDex 关键点轨迹做监督（无独立"state"流），可跳过本阶段。

若要保留（比如未来要把 EgoDex 轨迹喂给某个有独立"假想机器人state"的环境）：
1. 对齐前先平滑两路信号
2. 用互相关（cross-correlation）估计最优时间滞后
3. 在滞后对齐后的一阶差分上计算 **directional agreement (DA)**：
```python
def directional_agreement(state_diff, action_diff):
    same_sign = np.sign(state_diff) == np.sign(action_diff)
    return same_sign.mean()
```
4. DA 低于阈值（论文给的范围 0.6~0.7）则整段episode排除

### Stage 3: 极值过滤（Extreme Value Filtering）
```python
def extreme_value_mask(values, alpha=1.5):
    q01, q99 = np.percentile(values, [1, 99], axis=0)
    iqr = q99 - q01
    lower = q01 - alpha * iqr
    upper = q99 + alpha * iqr
    return (values >= lower) & (values <= upper)
```
- 按 embodiment 类型分别算分位数（这里 embodiment 统一是"人手"，但可以按任务类型/左右手分开算）
- **夹爪宽度维度豁免**（因为开合是双峰分布，不能用单峰假设的分位数裁剪）——对应这里的手部 `width` 信号同理，张开/抓握是双峰的，建议豁免或单独处理

### Stage 4: 关节-末端正向运动学一致性（仅适用于有机器人URDF的场景）
**这一步对纯人手数据（EgoDex）不直接适用**，因为人手没有URDF。但如果你的目标是"验证 retarget 后的虚拟夹爪轨迹在物理上自洽"，可以做一个简化版：
- 检查 `p_t`（手腕→末端的位移）与 `R_t` 是否在合理范围内联动变化（比如末端速度和角速度不应有荒谬的不连续跳变，这其实和 Stage 1 的检测有重叠）
- 如果后续要在真实夹爪机器人上回放这些轨迹，则需要做的是「轨迹的可达性检查」（IK是否有解），而不是 FK 一致性

### Stage 5: 基坐标系与朝向对齐
EgoDex 是相机坐标系下的数据（论文 Section 4.1: "All poses are represented in the camera frame"）。如果训练目标需要世界坐标系或某个固定参考系：
1. 用每帧的相机外参（extrinsics）把 camera frame 下的关键点变换到一个统一参考帧（如以 episode 第一帧的相机位姿为参考，或以 `hip`/`spine1` 为参考建立"躯干坐标系"）
2. 确保 +x 轴始终对应"人体朝向前方"（与论文里机器人"forward-facing"的约定类比）：可以用肩膀连线的法向量或头部朝向来定义

```python
def to_reference_frame(points_cam, cam_extrinsics, ref_extrinsics):
    # points_cam: (T,3) 在相机坐标系下的点
    # 转换到世界坐标系，再转换到参考帧
    points_world = transform_points(points_cam, cam_extrinsics)  # 用外参矩阵
    points_ref = transform_points(points_world, np.linalg.inv(ref_extrinsics))
    return points_ref
```

### Task 5: 把 Stage 1/3/5（EgoDex 真正需要的部分）写成 `filtering_pipeline.py`，输入一个 episode 的关键点序列，输出 `(keep_mask, corrected_keypoints)`。Stage 2/4 留空实现占位（带 TODO 注释和适用场景说明），方便未来接入混合数据集训练时启用。

---

### 4.1 当前实现约定：在 LeRobot v2.1 数据集上做过滤

第四步之后的过滤脚本以已经生成好的 LeRobot v2.1 数据集为输入，而不是回到原始 hdf5/npz 重新处理。这样可以复用同一套逻辑处理 EgoDex、人手数据和未来的机器人数据集：

- 核心过滤算法只接收 numpy 数组，例如 `action`、可选的 `observation.state`、可选的 confidence，不依赖 EgoDex 私有字段。
- LeRobot I/O 层只负责读取每个 episode 的 parquet、视频路径和 meta，并把过滤结果写成报告或新的 LeRobot 数据集。
- EgoDex 当前没有有效的 `observation.state`，所以 Stage 2 默认跳过；未来机械臂数据如果有真实 state，可以通过 `--state-key observation.state` 启用 state-action trend alignment。

当前 EgoDex 的 action 已经被定义在 episode 第一帧相机坐标系中，即 `episode_first_camera`。这个定义比“每帧当前相机系”更适合做 Stage 1/3/5：

- 参考系在一个 episode 内固定，position、velocity、acceleration、jerk 和极值统计都有稳定物理含义。
- 相机自身运动不会被混入 action 坐标系，突变检测不会因为坐标系跟着动而被掩盖。
- Stage 5 对 EgoDex 主要变成坐标系一致性检查和元数据约定；对机器人数据则可以检查或转换到 robot base/world 等固定参考系。

### 4.2 过滤输出策略：先报告，再生成新的 filtered dataset

过滤脚本不要原地修改输入的 LeRobot 数据集。推荐两阶段输出：

1. **质量报告与 mask**：先输出每个 episode 的诊断结果，供人工抽查和调阈值。
2. **新的 LeRobot 数据集**：确认阈值后，再把合格 episode 写入一个新的 filtered dataset 目录。

推荐报告目录结构：

```text
filter_reports/<dataset_name>/
├── filter_summary.json
├── episode_decisions.jsonl
├── flagged_frames.jsonl
└── frame_masks/
    ├── episode_000000.npz
    ├── episode_000001.npz
    └── ...
```

`episode_decisions.jsonl` 示例：

```json
{"episode_index": 0, "keep_episode": true, "bad_frame_ratio": 0.02, "reasons": []}
{"episode_index": 7, "keep_episode": false, "bad_frame_ratio": 0.41, "reasons": ["sudden_change_high", "low_confidence"]}
```

`frame_masks/episode_xxxxxx.npz` 至少保存：

- `keep_mask`
- `stage1_sudden_change_mask`
- `stage3_extreme_value_mask`
- `confidence_mask`（如果数据集中有 confidence）
- 各 stage 的 per-frame 或 per-dimension 统计量

### 4.3 为什么 filtered dataset 需要重新生成 meta

LeRobot v2.1 的 `meta/info.json`、`meta/episodes.jsonl`、`meta/episodes_stats.jsonl` 和 parquet 内的 `index/frame_index/episode_index` 是相互约束的：

- `info.json` 记录 `total_episodes`、`total_frames`、`total_videos`、`splits`、`total_chunks`。
- `episodes.jsonl` 记录每个 episode 的长度和任务。
- `episodes_stats.jsonl` 记录每个 episode 的统计量，训练时会被聚合成全局 stats。
- 每个 parquet 内部有连续的 `frame_index`、全局 `index`、`episode_index`、`task_index`。
- 视频路径也按 episode index 组织。

因此只要删除 episode 或删除帧，就不应该只改一个局部文件。最干净的方式是生成一个新的目录，例如：

```text
convert/output/egodex_demo_lerobot_v21_filtered/
```

并重新写：

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/<video_key>/episode_000000.mp4
meta/info.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/tasks.jsonl
```

初版过滤建议只做 **episode-level filtering**：如果一个 episode 的坏帧比例超过阈值，就整段丢弃；否则保留原 parquet 和原视频内容，并在新的 dataset 中重新编号 episode。这样可以避免视频和 parquet 时间戳错位。

如果未来要做 **frame-level filtering**，不能简单删除 parquet 里的坏帧并复制原 mp4，因为 LeRobot 根据 `timestamp` 从视频取帧，parquet 与视频会不一致。正确做法是把连续的 good frames 切成多个新 episode，并同步裁剪/重编码对应视频片段。



---

## 6. 建议的代码仓库结构

```
egodex_pipeline/
├── inspect_sample.py          # 第一步：探查原始数据schema
├── egodex_loader.py           # Task 1
├── extract_keypoints.py       # Task 2
├── action_alignment.py        # Task 3
├── smoothing.py                # Task 4
├── filtering_pipeline.py      # Task 5 (Stage 1/3/5实现, 2/4占位)
├── normalize.py                # 分位数归一化 [q01,q99] -> [-1,1]
├── fit_mano_from_keypoints.py # 可选，仅在需要跨数据集统一时实现
├── build_dataset.py            # 串联以上所有步骤，产出最终训练集(如LeRobot/HDF5格式)
├── tests/
│   ├── test_action_alignment.py   # 验证正交性、左右手对称性
│   ├── test_filtering.py
│   └── test_smoothing.py
└── configs/
    └── default.yaml            # 滤波窗口、阈值、是否启用MANO等超参数
```

## 7. 执行顺序建议（给 Codex 的 milestone）

1. `inspect_sample.py`：下载 test.zip，打印一个episode的完整字段结构，确认关节命名、confidence范围、fps、是否有language annotation字段
2. 实现 Task 1-3，写单元测试，跑通单个episode的可视化（画出 p_t, R_t 的轨迹，肉眼检查是否合理）
3. 实现 Task 4 平滑，对比平滑前后的轨迹噪声
4. 实现 Stage 1/3/5（Task 5），先在小样本上跑，统计被过滤掉的帧比例，调阈值
5. 实现 normalize.py
6. 串联成 build_dataset.py，跑全量或大样本，产出最终数据集
7. (可选，仅在确认需要时) 实现 MANO 拟合模块
