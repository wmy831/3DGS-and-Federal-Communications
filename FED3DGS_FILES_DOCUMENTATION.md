# Fed3DGS 项目文件功能说明

本文档详细说明 Fed3DGS 项目中每个文件的作用和功能。

## 📁 根目录文件

### `README.md`
- **功能**: 项目说明文档
- **内容**: 介绍 Fed3DGS 框架、安装步骤、数据集准备、训练和评估流程

### `LICENSE`
- **功能**: 项目许可证文件
- **说明**: 定义代码使用许可

### `environment.yml`
- **功能**: Conda 环境配置文件
- **内容**: 定义 Python 版本、PyTorch、CUDA 工具包等依赖项

---

## 📁 `gaussian-splatting/` - 核心训练和评估代码

### 主要训练/评估脚本

#### `train.py`
- **功能**: 训练本地 3D Gaussian Splatting 模型
- **作用**: 
  - 从 COLMAP 点云初始化 3D 高斯
  - 使用客户端图像数据训练局部模型
  - 优化高斯参数（位置、旋转、缩放、颜色、不透明度）
  - 支持断点续训（checkpoint）
- **输入**: 数据集路径、图像列表、COLMAP 结果
- **输出**: 训练好的本地模型（.ply 或 .pth）

#### `build_global_model.py`
- **功能**: 构建全局模型（联邦学习聚合）
- **作用**: 
  - 实现**蒸馏（Distillation）**机制
  - 从多个本地模型聚合生成全局模型
  - 使用本地模型渲染的目标图像来更新全局模型
  - 合并本地和全局的高斯点（concatenation）
  - 优化全局模型的 MLP（外观编码）和哈希编码
- **核心函数**: `distillation()` - 实现模型蒸馏
- **输入**: 本地模型目录、图像列表、数据集路径
- **输出**: `global_model.pth`（全局模型参数）

#### `eval.py`
- **功能**: 评估全局模型性能
- **作用**: 
  - 加载预训练的全局模型
  - 在验证集上渲染图像
  - 计算评估指标：**PSNR**、**SSIM**、**LPIPS**
  - 优化外观向量（appearance vector）以适配每个测试图像
- **输入**: 全局模型路径、数据集路径
- **输出**: `metrics.json`（包含所有指标）、渲染图像、深度图

---

### 📁 `scene/` - 场景和模型定义

#### `gaussian_model.py`
- **功能**: 定义 3D Gaussian 模型类
- **核心类**: 
  - `GaussianModel`: 管理所有高斯参数（位置、旋转、缩放、颜色、不透明度）
  - `HashEncoding`: 哈希网格编码（Instant-NGP 风格）
  - `MLP`: 多层感知机，用于外观编码
- **主要方法**:
  - `create_from_pcd()`: 从点云创建初始高斯
  - `training_setup()`: 设置优化器
  - `update_learning_rate()`: 学习率调度
  - `oneupSHdegree()`: 逐步增加球谐函数度数

#### `dataset_readers.py`
- **功能**: 读取不同格式的数据集
- **支持的格式**:
  - `readColmapSceneInfo()`: 读取 COLMAP 格式（sparse/0/）
  - `readNerfSyntheticInfo()`: 读取 Blender/NeRF 合成数据集
  - `readCamerasFromTransforms()`: 从 transforms.json 读取相机参数
- **输出**: `SceneInfo`（包含点云、相机信息、归一化参数）

#### `cameras.py`
- **功能**: 定义相机类
- **核心类**: `Camera`
- **作用**: 
  - 存储相机内参（焦距、主点）和外参（旋转、平移）
  - 计算投影矩阵
  - 管理图像数据

#### `colmap_loader.py`
- **功能**: 读取 COLMAP 二进制/文本格式
- **函数**:
  - `read_extrinsics_binary/text()`: 读取相机外参
  - `read_intrinsics_binary/text()`: 读取相机内参
  - `read_points3D_binary/text()`: 读取 3D 点云
  - `qvec2rotmat()`: 四元数转旋转矩阵

#### `__init__.py`
- **功能**: 场景初始化
- **作用**: 
  - 根据数据集类型自动选择读取器
  - 创建 `Scene` 对象，管理训练/测试相机和点云

---

### 📁 `utils/` - 工具函数

#### `model_update_utils.py`
- **功能**: 模型更新和渲染相关工具
- **主要函数**:
  - `rendering()`: 渲染 3D 高斯到 2D 图像
  - `get_model_params()`: 提取模型参数
  - `compute_visible_point_mask()`: 计算可见点掩码
  - `sample_cameras()`: 采样相机用于蒸馏
  - `get_cameras_from_metadata()`: 从 metadata 文件读取相机参数
  - `meganerf2colmap()`: 坐标系转换（Mega-NeRF ↔ COLMAP）
  - `faiss_knn()`: 使用 FAISS 进行 KNN 搜索
  - `knn_filtering()`: KNN 过滤
  - `reset_opacity()`: 重置不透明度

#### `graphics_utils.py`
- **功能**: 图形学工具函数
- **函数**:
  - `getProjectionMatrix()`: 计算投影矩阵
  - `fov2focal()` / `focal2fov()`: 视场角与焦距转换
  - `getWorld2View()`: 世界坐标到视图坐标变换
  - `geom_transform_points()`: 几何变换点云

#### `loss_utils.py`
- **功能**: 损失函数定义
- **函数**:
  - `l1_loss()`: L1 损失
  - `l2_loss()`: L2 损失
  - `ssim()`: 结构相似性指数（SSIM）

#### `image_utils.py`
- **功能**: 图像处理工具
- **函数**:
  - `mse()`: 均方误差
  - `psnr()`: 峰值信噪比

#### `sh_utils.py`
- **功能**: 球谐函数（Spherical Harmonics）工具
- **函数**:
  - `RGB2SH()`: RGB 颜色转球谐系数
  - `SH2RGB()`: 球谐系数转 RGB
  - `eval_sh()`: 评估球谐函数

#### `camera_utils.py`
- **功能**: 相机相关工具
- **函数**:
  - `loadCam()`: 加载相机
  - `cameraList_from_camInfos()`: 从相机信息创建相机列表
  - `camera_to_JSON()`: 相机信息序列化为 JSON

#### `general_utils.py`
- **功能**: 通用工具函数
- **函数**:
  - `inverse_sigmoid()`: 逆 sigmoid
  - `PILtoTorch()`: PIL 图像转 PyTorch 张量
  - `get_expon_lr_func()`: 指数学习率函数
  - `build_rotation()`: 构建旋转矩阵
  - `build_scaling_rotation()`: 构建缩放旋转矩阵
  - 四元数相关函数（旋转、插值等）

#### `system_utils.py`
- **功能**: 系统工具
- **函数**:
  - `mkdir_p()`: 递归创建目录
  - `searchForMaxIteration()`: 查找最大迭代次数

---

### 📁 `arguments/` - 参数定义

#### `__init__.py`
- **功能**: 定义命令行参数
- **类**:
  - `ModelParams`: 模型参数（SH 度数、点云路径等）
  - `PipelineParams`: 渲染管道参数
  - `OptimizationParams`: 优化参数（学习率、迭代次数等）

---

### 📁 `gaussian_renderer/` - 渲染器

#### `__init__.py`
- **功能**: 导出渲染函数
- **函数**: `render()` - 主要渲染接口

#### `network_gui.py`
- **功能**: 网络 GUI（用于实时可视化训练过程）

---

### 📁 `lpipsPyTorch/` - LPIPS 指标

#### `modules/lpips.py`
- **功能**: LPIPS（Learned Perceptual Image Patch Similarity）实现
- **作用**: 使用预训练 VGG 网络计算感知相似度

#### `modules/networks.py`
- **功能**: 定义 LPIPS 使用的网络架构（VGG16 等）

#### `modules/utils.py`
- **功能**: LPIPS 工具函数

---

### 📁 `submodules/` - 第三方子模块

#### `diff-gaussian-rasterization/`
- **功能**: 可微分的 3D 高斯光栅化（CUDA 实现）
- **作用**: 
  - 将 3D 高斯投影到 2D 图像
  - 支持前向和反向传播
  - 核心 CUDA 代码：`forward.cu`, `backward.cu`, `rasterizer_impl.cu`
- **编译**: 需要 CUDA 和 C++ 编译器

#### `simple-knn/`
- **功能**: 简单的 KNN 搜索（CUDA 实现）
- **作用**: 快速计算点云中最近邻距离（用于初始化高斯缩放）

---

## 📁 `tools/` - 数据处理工具

#### `gen_client_data.py`
- **功能**: 生成客户端数据列表
- **作用**: 
  - 从数据集中随机采样图像，分配给不同客户端
  - 使用空间聚类（基于相机位置）确保客户端数据有空间相关性
  - 输出每个客户端的图像列表文件（.txt）
- **输入**: 数据集路径、客户端数量
- **输出**: 图像列表目录（每个客户端一个 .txt 文件）

#### `create_db.py`
- **功能**: 创建 COLMAP 数据库
- **作用**: 
  - 从 metadata (.pt) 文件读取相机参数
  - 将相机参数写入 COLMAP 数据库格式
  - 为 COLMAP 三角化准备数据
- **输入**: 数据集路径、图像列表
- **输出**: COLMAP 数据库文件（database.db）

#### `triangulate_colmap.sh` / `triangulate_colmap.bat`
- **功能**: COLMAP 三角化脚本
- **作用**: 
  - 调用 COLMAP 进行特征提取、匹配、三角化
  - 生成初始 3D 点云（sparse/0/points3D.ply）
  - 使用已知的相机参数（从 metadata），只做三角化
- **步骤**:
  1. 创建数据库（`create_db.py`）
  2. 特征提取（`colmap feature_extractor`）
  3. 特征匹配（`colmap exhaustive_matcher`）
  4. 点三角化（`colmap point_triangulator`）

#### `merge_val_train.py`
- **功能**: 合并验证集和训练集
- **作用**: 
  - 将验证集数据复制到训练集（Mega-NeRF 设置）
  - 验证图像的左半部分用于训练
- **操作**: `copy`（合并）或 `remove`（移除）

#### `database.py`
- **功能**: COLMAP 数据库操作
- **作用**: 
  - 封装 SQLite 数据库操作
  - 添加相机、图像、特征点等
  - 读取/写入 COLMAP 数据库格式

#### `read_write_model.py`
- **功能**: 读取/写入 COLMAP 模型文件
- **作用**: 
  - 读取 COLMAP 的 cameras.bin/txt, images.bin/txt, points3D.bin/txt
  - 写入 COLMAP 格式文件
  - 支持二进制和文本格式

---

## 📁 `scripts/` - 训练脚本

#### `client_training.sh` / `client_training.bat`
- **功能**: 批量训练客户端模型
- **作用**: 
  - 循环训练多个客户端（从 start_idx 到 end_idx）
  - 对每个客户端：
    1. 运行 COLMAP 三角化（`triangulate_colmap.sh/bat`）
    2. 训练本地模型（`train.py`）
  - 保存 COLMAP 结果和训练好的模型
- **参数**: 起始索引、结束索引、COLMAP 目录、数据集根目录、图像列表目录、输出目录

---

## 📁 `outputs/` - 输出目录

- **功能**: 存储训练和评估结果
- **内容**: 
  - 训练日志（console.log）
  - 模型检查点
  - 评估指标（metrics.json）
  - 渲染图像

### `metrics.json` - 评估指标文件

- **生成位置**: 由 `eval.py` 在输出目录中生成
- **文件格式**: JSON 格式
- **包含指标**:
  ```json
  {
    "psnr": <float>,  // 峰值信噪比（Peak Signal-to-Noise Ratio）
    "ssim": <float>,  // 结构相似性指数（Structural Similarity Index）
    "lpips": <float>  // 学习感知图像块相似度（Learned Perceptual Image Patch Similarity）
  }
  ```

- **指标说明**:
  - **PSNR** (Peak Signal-to-Noise Ratio)
    - 范围: 通常 0-50+ dB，越高越好
    - 含义: 衡量图像像素级误差，基于均方误差（MSE）
    - 计算: `psnr = 10 * log10(MAX² / MSE)`
    - 典型值: 20-30 dB 为可接受，30+ dB 为优秀
  
  - **SSIM** (Structural Similarity Index)
    - 范围: 0-1，越接近 1 越好
    - 含义: 衡量图像结构相似性，考虑亮度、对比度、结构
    - 计算: 基于局部窗口的亮度、对比度、结构比较
    - 典型值: 0.8+ 为良好，0.9+ 为优秀
  
  - **LPIPS** (Learned Perceptual Image Patch Similarity)
    - 范围: 0-1，越小越好（0 表示完全相同）
    - 含义: 基于深度学习的感知相似度，更符合人类视觉感知
    - 计算: 使用预训练 VGG16 网络提取特征，计算特征距离
    - 典型值: <0.1 为优秀，<0.2 为良好

- **生成过程**:
  1. 对验证集中的每张图像：
     - 优化外观向量（appearance vector）以适配该图像
     - 渲染图像（使用优化后的外观向量）
     - 计算该图像的 PSNR、SSIM、LPIPS
  2. 计算所有图像的平均值
  3. 保存平均值到 `metrics.json`

- **示例内容**:
  ```json
  {
    "psnr": 28.456,
    "ssim": 0.892,
    "lpips": 0.156
  }
  ```

- **相关文件**:
  - `console.log`: 包含每张图像的详细评估日志
  - 渲染图像: 保存在输出目录中（与验证图像同名）
  - 深度图: 保存在输出目录中（前缀 `depth-`）

---

## 🔄 工作流程总结

### 1. 数据准备阶段
```
merge_val_train.py → gen_client_data.py
```

### 2. 本地模型训练阶段
```
triangulate_colmap.sh/bat → train.py
(对每个客户端重复)
```

### 3. 全局模型构建阶段
```
build_global_model.py
(聚合所有本地模型)
```

### 4. 评估阶段
```
eval.py
(在验证集上评估全局模型)
```

---

## 📝 关键概念

- **3D Gaussian Splatting**: 使用 3D 高斯点表示场景，通过可微分光栅化渲染
- **联邦学习**: 多个客户端协作训练，不共享原始数据
- **蒸馏（Distillation）**: 使用本地模型渲染的目标图像来更新全局模型
- **球谐函数（SH）**: 用于表示方向相关的颜色
- **哈希编码**: Instant-NGP 风格的快速特征编码
- **COLMAP**: Structure-from-Motion 工具，用于生成初始点云

---

## 🔧 依赖关系

- **PyTorch**: 深度学习框架
- **COLMAP**: 3D 重建工具
- **FAISS**: 快速相似性搜索
- **tiny-cuda-nn**: CUDA 神经网络库（用于哈希编码）
- **CUDA**: GPU 加速（必需）

---

*最后更新: 2024*

