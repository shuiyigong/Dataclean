# Egodata preprocess
## Installation and Code Structure
```
conda create --name ego python==3.11
conda activate ego
conda install -c conda-forge ffmpeg=7.1.1
pip install -r requirements.txt
```
### EgoDex:
```
python convert/build_egodex_prefilter_step1.py 
#把人手映射成夹爪，存成npz文件

python convert/visualize_script/visualize_gripper_axes_egodex.py
#可视化夹爪坐标系，附带arkit confidence

python convert/egodex_npz_to_lerobot_v21_step2.py
#把npz文件转成lerobotv2.1格式
```

### EgoVerse:
```
python /mnt/project_rlinf/runze/ml-egodex/convert/build_egodex_prefilter_step1.py 
#把人手映射成夹爪，存成npz文件

python convert/visualize_script/visualize_gripper_axes_egoverse.py
#可视化夹爪坐标系

python convert/egoverse_npz_to_lerobot_v21_step2.py
#把npz文件转成lerobotv2.1格式
```

## Lerobot v2.1 Feature
```
observation.state: ( position(3) + rotation(6) + gripper(1) ) * 2, all in first-frame camera frame, same with action  
action: ( position(3) + rotation(6) + gripper(1) ) * 2, all in first-frame camera frame  
observation.confidence(egodex only): confidence of arkit joint of "left_wrist", "left_thumb", "left_index", "left_middle", "right_wrist", "right_thumb", "right_index", "right_middle"   
observation.camera_extrinsics_world  
observation.camera_intrinsics  
observation.images.camera_top  
```

# Qwen Data Pipeline
```
见/mnt/project_rlinf/runze/ml-egodex/qwen-manip-preprocess
```