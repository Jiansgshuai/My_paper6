# 项目文件索引

## 📂 完整文件列表

### 代码实现（code/目录）

| 文件名 | 说明 | 行数 | 主要内容 |
|--------|------|------|---------|
| **README.md** | 代码说明文档 | 120 | 项目概述、安装、使用方法 |
| **requirements.txt** | Python依赖包 | 15 | numpy, open3d, sklearn等 |
| **innovation1_svo_background_removal.py** | 创新1实现 | 380 | SVO数据结构、背景去除算法、演示 |
| **innovation2_temporal_association.py** | 创新2实现 | 450 | L-shape拟合、跨帧关联、轨迹管理 |
| **innovation3_lane_vehicle_classification.py** | 创新3实现 | 420 | 车道划分、SVM分类、特征提取 |
| **demo.py** | 完整流程演示 | 300 | 集成三个创新点的完整pipeline |

### 工具包（code/utils/目录）

| 文件名 | 说明 | 行数 | 主要功能 |
|--------|------|------|---------|
| **__init__.py** | 包初始化 | 40 | 导出所有工具函数 |
| **point_cloud_utils.py** | 点云处理工具 | 200 | PCD读写、下采样、聚类、边界框 |
| **visualization.py** | 可视化工具 | 180 | 2D/3D轨迹绘图、速度曲线、混淆矩阵 |
| **evaluation.py** | 评估指标 | 220 | MOTA/MOTP/IDF1、分类指标、速度统计 |

### LaTeX论文文件

| 文件名 | 说明 | 字数 | 主要章节 |
|--------|------|------|---------|
| **paper_main.tex** | 主论文文件 | 8,000 | 完整论文包含三个创新点 |
| **paper_algorithms.tex** | 算法伪代码 | 1,500 | 6个算法的伪代码 |
| **trb_template.bib** | 参考文献 | 26条 | BibTeX格式引用 |

### 文档说明

| 文件名 | 说明 | 字数 | 用途 |
|--------|------|------|------|
| **INNOVATIONS_IMPLEMENTATION.md** | 创新点详细技术文档 | 6,000 | 方法原理、实现细节、实验结果 |
| **LATEX_README.md** | LaTeX编译指南 | 2,500 | 编译方法、修改建议、问题排查 |
| **PROJECT_SUMMARY.md** | 项目总结 | 3,500 | 整体概览、成果汇总、后续计划 |
| **INDEX.md** | 本文件 | 500 | 文件索引和快速导航 |

---

## 🎯 快速导航

### 如果你想...

#### 运行代码
1. 阅读：[code/README.md](code/README.md)
2. 安装依赖：`pip install -r code/requirements.txt`
3. 运行演示：`python code/demo.py`

#### 了解方法原理
1. 阅读：[INNOVATIONS_IMPLEMENTATION.md](INNOVATIONS_IMPLEMENTATION.md)
2. 查看章节：
   - 创新1：SVO背景去除
   - 创新2：跨帧关联
   - 创新3：车道与分类

#### 编译论文
1. 阅读：[LATEX_README.md](LATEX_README.md)
2. 编译命令：`latexmk paper_main.tex -pdf -pvc`
3. 查看输出：paper_main.pdf

#### 查看实验结果
1. 运行代码生成图片到results/目录
2. 查看论文中的表格数据
3. 阅读PROJECT_SUMMARY.md中的结果汇总

#### 修改和扩展
1. 代码扩展：查看code/目录中的Python文件
2. 论文修改：编辑paper_main.tex
3. 添加实验：参考demo.py中的流程

---

## 📊 代码模块依赖关系

```
demo.py (完整流程)
    ↓
├── innovation1_svo_background_removal.py
│   └── utils/point_cloud_utils.py
│
├── innovation2_temporal_association.py
│   └── utils/point_cloud_utils.py
│
└── innovation3_lane_vehicle_classification.py
    ├── utils/point_cloud_utils.py
    └── utils/evaluation.py

可视化：utils/visualization.py
评估：utils/evaluation.py
```

---

## 📄 论文章节对应代码

| 论文章节 | 对应代码文件 | 关键类/函数 |
|---------|------------|-----------|
| Section 3.1: SVO背景去除 | innovation1_svo_background_removal.py | `SparseVoxelOctree`, `SVOBackgroundRemover` |
| Section 3.2: 跨帧关联 | innovation2_temporal_association.py | `TemporalAssociator`, `LShapeFitting` |
| Section 3.3: 车道与分类 | innovation3_lane_vehicle_classification.py | `SpatialLaneDivision`, `SVMVehicleClassifier` |
| Algorithm 1 | paper_algorithms.tex 第12行 | SVO背景去除伪代码 |
| Algorithm 2 | paper_algorithms.tex 第35行 | L-shape拟合伪代码 |
| Algorithm 3 | paper_algorithms.tex 第60行 | 跨帧关联伪代码 |
| Table 1 | paper_main.tex 第452行 | 背景去除性能表格 |
| Table 2 | paper_main.tex 第472行 | 轨迹关联性能表格 |
| Table 3 | paper_main.tex 第492行 | 分类准确率表格 |

---

## 🔑 关键函数速查

### 创新1：背景去除

```python
# 主要类
class SparseVoxelOctree:
    def __init__(bounds, max_depth, leaf_size, background_threshold)
    def insert(point, node, depth)
    def mark_background(node)
    def is_background_point(point)

class SVOBackgroundRemover:
    def process_frame(pcd)
    def process_sequence(pcd_list)
```

### 创新2：跨帧关联

```python
# 主要类
class TemporalAssociator:
    def process_frame(observations)
    def predict_next_position(dt)
    def estimate_motion_direction()

class LShapeFitting:
    @staticmethod
    def fit_lshape(points)
```

### 创新3：车道与分类

```python
# 主要类
class SpatialLaneDivision:
    def divide_lanes(trajectories)

class SVMVehicleClassifier:
    def train(features, labels)
    def predict(features)

class VehicleFeatureExtractor:
    @staticmethod
    def extract_features(points, bbox)
```

---

## 📈 数据流图

```
原始点云序列 (PCD文件)
    ↓
[创新1] SVOBackgroundRemover
    ↓
前景点云（车辆点云）
    ↓
DBSCAN聚类
    ↓
车辆观测（VehicleObservation）
    ↓
[创新2] TemporalAssociator
    ↓
连续轨迹（VehicleTrack）
    ↓
[创新3] SpatialLaneDivision + SVMVehicleClassifier
    ↓
增强轨迹数据（EnrichedTrajectory）
    - 位置、速度
    - 车道ID
    - 车型标签
```

---

## 🎨 生成的可视化文件

运行代码后会在results/目录生成：

1. **innovation2_trajectories.png**
   - 尺寸：15×5英寸
   - 内容：3个子图（XY轨迹、3D轨迹、速度曲线）
   - 用途：展示跨帧关联效果

2. **innovation3_lane_classification.png**
   - 尺寸：16×6英寸
   - 内容：2个子图（车道划分、车型分类）
   - 用途：展示车道和车型识别

3. **complete_pipeline_results.png**
   - 尺寸：18×10英寸
   - 内容：4个子图（俯视图、分类图、3D图、统计）
   - 用途：完整流程结果展示

---

## 🔧 配置参数速查

### SVO背景去除参数

```python
voxel_size = 0.5              # 体素大小（米）
background_threshold = 5       # 背景判定阈值（帧）
sor_nb_neighbors = 20         # SOR邻域点数
sor_std_ratio = 2.0           # SOR标准差倍数
```

### 跨帧关联参数

```python
max_distance = 3.0            # 最大关联距离（米）
max_missing_frames = 5        # 最大丢失帧数
angle_weight = 0.3            # 角度权重
dt = 0.1                      # 时间间隔（秒）
```

### 车道与分类参数

```python
# 车道划分
cluster_eps = 2.0             # DBSCAN半径（米）
num_lanes = 3                 # 车道数量（None为自动）

# SVM分类
kernel = 'rbf'                # 核函数
C = 1.0                       # 惩罚参数
gamma = 'scale'               # 核参数
```

---

## 📚 推荐阅读顺序

### 初次接触项目
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 了解整体
2. [code/README.md](code/README.md) - 运行演示
3. [INNOVATIONS_IMPLEMENTATION.md](INNOVATIONS_IMPLEMENTATION.md) - 深入原理

### 准备修改代码
1. [code/innovation1_svo_background_removal.py](code/innovation1_svo_background_removal.py)
2. [code/innovation2_temporal_association.py](code/innovation2_temporal_association.py)
3. [code/innovation3_lane_vehicle_classification.py](code/innovation3_lane_vehicle_classification.py)
4. [code/utils/](code/utils/) - 工具函数

### 准备写论文
1. [LATEX_README.md](LATEX_README.md) - 编译指南
2. [paper_main.tex](paper_main.tex) - 论文主体
3. [paper_algorithms.tex](paper_algorithms.tex) - 算法伪代码
4. [trb_template.bib](trb_template.bib) - 参考文献

---

## ✅ 检查清单

### 代码运行前
- [ ] 已安装Python 3.8+
- [ ] 已安装所有依赖包
- [ ] 已创建results/目录

### 论文编译前
- [ ] 已安装LaTeX环境
- [ ] trbunofficial.cls文件存在
- [ ] trb_template.bib文件完整
- [ ] 已更新作者信息

### 投稿准备
- [ ] 代码运行无误
- [ ] 论文编译成功
- [ ] 图表清晰美观
- [ ] 参考文献格式正确
- [ ] 已通过查重检查

---

## 🆘 常见问题

**Q: 代码运行报错找不到模块？**  
A: 确保在code/目录下运行，或将code/添加到PYTHONPATH

**Q: LaTeX编译失败？**  
A: 查看LATEX_README.md的问题排查部分

**Q: 如何使用自己的数据？**  
A: 参考demo.py中的数据加载部分，准备PCD格式点云

**Q: 如何调整参数？**  
A: 查看各文件的`__init__`方法中的参数说明

---

**最后更新**：2026年6月28日  
**版本**：1.0  
**维护者**：项目作者
