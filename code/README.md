# 路侧激光雷达车辆轨迹采集三大创新方法实现

## 项目概述
本项目实现了学位论文《考虑车辆形态位姿一致性的路侧激光雷达车辆轨迹采集》中的三个核心方法创新。

## 三大创新点

### 1. 基于稀疏体素八叉树（SVO）的增量式时空背景滤除
- **文件**: `innovation1_svo_background_removal.py`
- **核心技术**: 稀疏体素八叉树、GPU加速、统计离群点滤波（SOR）
- **优势**: 高效背景分割、适应动态场景

### 2. 基于时空质心与运动主方向的跨帧关联
- **文件**: `innovation2_temporal_association.py`
- **核心技术**: 时空质心分析、运动主方向估计、L-shape拟合
- **优势**: 解决轨迹中断、保持车辆形态位姿一致性

### 3. 基于点云空间分布的车道划分与车型分类
- **文件**: `innovation3_lane_vehicle_classification.py`
- **核心技术**: 点云空间分析、SVM分类器、无监督车道划分
- **优势**: 无需车道线标注、多维度轨迹数据

## 项目结构
```
code/
├── README.md                                   # 项目说明
├── requirements.txt                            # 依赖包
├── innovation1_svo_background_removal.py       # 创新1: SVO背景滤除
├── innovation2_temporal_association.py         # 创新2: 跨帧关联
├── innovation3_lane_vehicle_classification.py  # 创新3: 车道与车型分类
├── utils/
│   ├── __init__.py
│   ├── point_cloud_utils.py                   # 点云处理工具
│   ├── visualization.py                        # 可视化工具
│   └── evaluation.py                           # 评估指标
├── data/
│   └── sample_data/                            # 示例数据
├── results/                                    # 结果输出
└── demo.py                                     # 完整流程演示
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 单独运行各创新点
```python
# 创新1: 背景滤除
python innovation1_svo_background_removal.py --input data/sample.pcd

# 创新2: 跨帧关联
python innovation2_temporal_association.py --input data/frames/

# 创新3: 车道与车型分类
python innovation3_lane_vehicle_classification.py --input data/trajectories.pkl
```

### 运行完整流程
```python
python demo.py --input data/raw_pointcloud_sequence/
```

## 数据格式
- 输入: PCD格式点云文件或numpy数组
- 输出: 车辆轨迹数据（包含位置、速度、车型、车道信息）

## 性能指标
详见各模块的评估报告和可视化结果。
