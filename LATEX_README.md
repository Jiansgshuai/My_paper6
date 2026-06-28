# LaTeX论文文件说明

## 📄 文件列表

- **paper_main.tex** - 完整论文主文件（包含三个创新点的详细方法）
- **paper_algorithms.tex** - 算法伪代码补充文件
- **Unofficial_Transportation_Research_Board__TRB__LaTeX_template/** - TRB模板文件夹

## 📝 论文结构

### 主要章节

1. **Abstract（摘要）**
   - 结构化摘要，包含：Objectives, Methods, Findings, Novelty, Practical Applications

2. **Introduction（引言）**
   - 研究背景
   - 问题陈述
   - 研究动机

3. **Literature Review（文献综述）**
   - 背景去除方法
   - 多目标跟踪方法
   - 车道检测与车型分类方法

4. **Methodology（方法论）** - 核心章节
   - **Innovation 1: SVO-Based Incremental Background Removal**
     - 稀疏体素八叉树结构
     - 增量式背景建模
     - 前景提取算法
   
   - **Innovation 2: Cross-Frame Association with Spatio-Temporal Centroids**
     - 车辆观测表示
     - L-shape拟合算法
     - 跨帧关联策略
   
   - **Innovation 3: Spatial Distribution-Based Lane Division and Classification**
     - 无监督车道划分
     - 多维特征提取
     - SVM车型分类

5. **Experimental Results（实验结果）**
   - 实验设置
   - 背景去除性能
   - 轨迹关联结果
   - 车道划分与分类准确率
   - 整体系统性能

6. **Discussion（讨论）**
   - 方法优势
   - 局限性与未来工作

7. **Conclusion（结论）**

## 🔧 编译方法

### 方法1：使用latexmk（推荐）

```bash
cd d:\研究\paper6\My_paper6
latexmk paper_main.tex -pdf -pvc
```

### 方法2：手动编译

```bash
pdflatex paper_main.tex
bibtex paper_main
pdflatex paper_main.tex
pdflatex paper_main.tex
```

### 方法3：使用Overleaf

1. 上传所有文件到Overleaf项目
2. 确保包含TRB模板类文件（trbunofficial.cls）
3. 点击"Recompile"

## 📦 所需LaTeX包

确保安装了以下包：
- amsmath - 数学公式
- algorithm, algorithmic - 算法伪代码
- booktabs - 表格美化
- graphicx - 图片插入
- hyperref - 超链接
- lineno - 行号

## 🎨 算法伪代码使用

如果要在主文件中添加算法伪代码，在paper_main.tex的preamble中添加：

```latex
\usepackage{algorithm}
\usepackage{algorithmic}
```

然后在相应位置插入paper_algorithms.tex中的算法块。

## 📊 三大创新点数学公式总结

### 创新1：八叉树背景去除

**八叉象限计算**：
```
octant(p, c) = 4·I[p_x ≥ c_x] + 2·I[p_y ≥ c_y] + I[p_z ≥ c_z]
```

**背景标记**：
```
f_bg = true if n_visit ≥ θ_bg
```

### 创新2：跨帧关联

**L-shape拟合目标**：
```
θ* = argmin_θ A(θ)
A(θ) = (max p_x(θ) - min p_x(θ)) · (max p_y(θ) - min p_y(θ))
```

**关联代价**：
```
C_ij = d_pos + λ · d_ang
d_pos = ||ĉ_t^i - c_t^j||
d_ang = arccos(d_i · d_j^obs) / π
```

### 创新3：车道与车型分类

**特征向量**：
```
f = [l, w, h, V, r, ρ, z_c, d̄, σ_d, σ_z, Δz]^T
```

**SVM核函数**：
```
K(f_i, f_j) = exp(-γ ||f_i - f_j||²)
```

## 📈 实验结果数据

### 背景去除
- 去除率: 85-90%
- 车辆点保留率: >95%
- 处理速度: 0.15秒/帧

### 轨迹关联
- 轨迹完整率: 85-90%
- ID切换次数: 0
- 处理速度: 0.05秒/帧

### 车道与分类
- 车道识别准确率: 100%
- 车型分类准确率: 85-95%

## 🔄 修改建议

### 1. 更新作者信息

在paper_main.tex中修改：

```latex
\TRBauthor{您的姓名}{您的单位}{您的邮箱}[城市，省份，国家]
\TRBauthor{第二作者}{单位}{邮箱}[地址]
```

### 2. 添加参考文献

创建trb_template.bib文件，添加引用：

```bibtex
@article{example2023,
  author = {Zhang, Wei and Li, Ming},
  title = {Vehicle Tracking Using LiDAR},
  journal = {Transportation Research Part C},
  year = {2023},
  volume = {150},
  pages = {104--120}
}
```

### 3. 插入图片

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\linewidth]{your_figure.png}
  \caption{图片说明.}\label{fig:yourfig}
\end{figure}
```

### 4. 修改实验数据

在paper_main.tex的Table部分更新您的真实实验数据。

## 🚀 快速开始检查清单

- [ ] 安装LaTeX环境（TeX Live或MiKTeX）
- [ ] 确认trbunofficial.cls文件存在
- [ ] 更新作者信息
- [ ] 添加参考文献到.bib文件
- [ ] 准备图片文件（PNG或PDF格式）
- [ ] 更新实验数据表格
- [ ] 运行编译命令
- [ ] 检查生成的PDF文件

## 📞 问题排查

### 编译错误："trbunofficial.cls not found"

**解决方法**：
1. 确保trbunofficial.cls与paper_main.tex在同一目录
2. 或将其放到LaTeX搜索路径中

### 参考文献不显示

**解决方法**：
1. 确保trb_template.bib存在
2. 运行完整编译序列：pdflatex → bibtex → pdflatex → pdflatex

### 算法伪代码报错

**解决方法**：
```bash
# 安装algorithm包
tlmgr install algorithms
```

## 📚 TRB投稿要求

根据TRB 2027年会要求：

1. **页数限制**：通常6-8页
2. **摘要**：结构化摘要，<300词
3. **格式**：双栏、行号、1英寸边距
4. **引用格式**：Chicago author-date格式
5. **图表**：粗体标题

当前模板已按照这些要求配置。

## 🎯 下一步工作

1. **完善实验**：使用真实数据集（KITTI、nuScenes等）
2. **添加对比实验**：与baseline方法对比
3. **补充可视化**：添加轨迹图、混淆矩阵等
4. **完善参考文献**：至少20-30篇相关文献
5. **请同事审阅**：获取反馈意见

## 📖 相关资源

- TRB官方指南：https://trb.secure-platform.com/a/page/TRBPaperReview
- LaTeX教程：https://www.overleaf.com/learn
- 模板GitHub：https://github.com/chiehrosswang/TRB_LaTeX_tex

---

**最后更新**：2026年6月28日  
**联系方式**：[您的邮箱]
