# Fed3DGS 训练与评估指令汇总

> 作者：小多子 😊  
> 日期：2025年12月3日

---

## 📋 **目录**

1. [训练本地模型](#1-训练本地模型)
2. [构建全局模型](#2-构建全局模型)
3. [评估全局模型](#3-评估全局模型)
4. [评估本地模型](#4-评估本地模型)
5. [辅助工具](#5-辅助工具)
6. [常见问题](#6-常见问题)

---

## 1. 训练本地模型

### **基础命令：**
```powershell
scripts\client_training.bat <起始索引> <结束索引> <COLMAP输出目录> <数据集根目录> <图像列表目录> <输出目录>
```

### **示例：**

#### **训练 3 个客户端（00000, 00001, 00002）**
```powershell
scripts\client_training.bat 0 2 D:\githubdownloads\Fed3DGS_data\colmap-results\rubble D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble D:\githubdownloads\Fed3DGS_data\local-models\rubble
```

#### **训练 10 个客户端（00000-00009）**
```powershell
scripts\client_training.bat 0 9 D:\githubdownloads\Fed3DGS_data\colmap-results\rubble D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble D:\githubdownloads\Fed3DGS_data\local-models\rubble
```

#### **增量训练（继续训练更多客户端）**
```powershell
# 假设已有 0-9，继续训练 10-19
scripts\client_training.bat 10 19 D:\githubdownloads\Fed3DGS_data\colmap-results\rubble D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble D:\githubdownloads\Fed3DGS_data\local-models\rubble
```

### **参数说明：**
- `0 2`：训练客户端编号从 0 到 2（包含0和2，共3个）
- **COLMAP输出目录**：存放三角化结果（相机参数、点云）
- **数据集根目录**：包含 `train/rgbs` 和 `train/metadata` 的目录
- **图像列表目录**：包含每个客户端的图像列表文件（00000.txt, 00001.txt, ...）
- **输出目录**：存放训练好的模型

### **训练时长估算：**
- 每个客户端：约 30-60 分钟
- 10 个客户端：约 5-10 小时

### **输出文件：**
```
<输出目录>/
├── 00000/
│   └── point_cloud/
│       └── iteration_20000/
│           └── point_cloud.ply  # 训练好的模型
├── 00001/
│   └── ...
└── 00002/
    └── ...
```

---

## 2. 构建全局模型

### **基础命令：**
```powershell
python gaussian-splatting/build_global_model.py -w -o <输出目录> -m <本地模型目录> -i <图像列表目录> -data <数据集目录> --sh-degree <度数> --n-clients <客户端数量>
```

### **示例：**

#### **使用 3 个客户端构建**
```powershell
python gaussian-splatting/build_global_model.py -w -o D:\githubdownloads\Fed3DGS_data\global-models\rubble -m D:\githubdownloads\Fed3DGS_data\local-models\rubble -i D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 --n-clients 3
```

#### **使用 10 个客户端构建（推荐）**
```powershell
python gaussian-splatting/build_global_model.py -w -o D:\githubdownloads\Fed3DGS_data\global-models\rubble-10clients -m D:\githubdownloads\Fed3DGS_data\local-models\rubble -i D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 --n-clients 10
```

#### **使用更保守的学习率（避免 NaN）**
```powershell
python gaussian-splatting/build_global_model.py -w -o D:\githubdownloads\Fed3DGS_data\global-models\rubble-stable -m D:\githubdownloads\Fed3DGS_data\local-models\rubble -i D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 --n-clients 10 --lr-opacity 0.01 --lr-mlp 1e-5
```

#### **使用安全版本（自动跳过缺失模型）**
```powershell
python gaussian-splatting/build_global_model_safe.py -w -o D:\githubdownloads\Fed3DGS_data\global-models\rubble-safe -m D:\githubdownloads\Fed3DGS_data\local-models\rubble -i D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3
```

### **关键参数说明：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `-w` | ✅ | 白色背景（必须与训练一致） |
| `-o` | ✅ | 输出目录 |
| `-m` | ✅ | 本地模型目录 |
| `-i` | ✅ | 图像列表目录 |
| `-data` | ✅ | 数据集根目录 |
| `--sh-degree` | ✅ | **必须与训练时一致**（默认3） |
| `--n-clients` | 推荐 | 使用的客户端数量（默认-1=全部） |
| `--lr-opacity` | 可选 | 不透明度学习率（默认0.05，建议0.01） |
| `--lr-mlp` | 可选 | MLP学习率（默认1e-4，建议1e-5） |

### **输出文件：**
```
<输出目录>/
├── global_model.pth              # 最终全局模型 ⭐
├── global_model_0100clients.pth  # 中间保存点（每100个客户端）
├── global_model_0200clients.pth
└── console.log                   # 训练日志
```

---

## 3. 评估全局模型

### **基础命令：**
```powershell
python gaussian-splatting/eval.py -w -o <输出目录> -g <全局模型路径> -data <数据集目录> --sh-degree <度数> -r <分辨率缩放>
```

### **示例：**

#### **标准评估**
```powershell
python gaussian-splatting/eval.py -w -o D:\githubdownloads\Fed3DGS_data\outputs\rubble -g D:\githubdownloads\Fed3DGS_data\global-models\rubble\global_model.pth -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 -r 8
```

#### **高分辨率评估（需要更多显存）**
```powershell
python gaussian-splatting/eval.py -w -o D:\githubdownloads\Fed3DGS_data\outputs\rubble-highres -g D:\githubdownloads\Fed3DGS_data\global-models\rubble\global_model.pth -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 -r 4
```

#### **低分辨率快速评估**
```powershell
python gaussian-splatting/eval.py -w -o D:\githubdownloads\Fed3DGS_data\outputs\rubble-lowres -g D:\githubdownloads\Fed3DGS_data\global-models\rubble\global_model.pth -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 -r 16
```

### **关键参数说明：**
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `-w` | 白色背景（与训练一致） | 必须 |
| `-g` | 全局模型路径（.pth） | 必须 |
| `--sh-degree` | **必须与训练一致** | 3 |
| `-r` | 分辨率缩放倍数 | 8 或 16（显存不足时） |

### **输出文件：**
```
<输出目录>/
├── metrics.json         # 📊 评估指标（PSNR, SSIM, LPIPS）⭐
├── console.log          # 详细日志
├── 000000.jpg          # 渲染图像
├── 000083.jpg
├── ...
├── depth-000000.jpg    # 深度图
└── depth-000083.jpg
```

### **查看评估结果：**
```powershell
# 查看指标
Get-Content D:\githubdownloads\Fed3DGS_data\outputs\rubble\metrics.json | ConvertFrom-Json

# 或
notepad D:\githubdownloads\Fed3DGS_data\outputs\rubble\metrics.json
```

---

## 4. 评估本地模型

### **基础命令：**
```powershell
python gaussian-splatting/eval.py -w -o <输出目录> -g <本地模型PLY路径> -data <数据集目录> --sh-degree <度数> -r <分辨率缩放>
```

### **示例：**

#### **评估客户端 00000**
```powershell
python gaussian-splatting/eval.py -w -o D:\githubdownloads\Fed3DGS_data\outputs\local-00000 -g D:\githubdownloads\Fed3DGS_data\local-models\rubble\00000\point_cloud\iteration_20000\point_cloud.ply -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 -r 8
```

#### **评估客户端 00001**
```powershell
python gaussian-splatting/eval.py -w -o D:\githubdownloads\Fed3DGS_data\outputs\local-00001 -g D:\githubdownloads\Fed3DGS_data\local-models\rubble\00001\point_cloud\iteration_20000\point_cloud.ply -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 -r 8
```

### **注意事项：**
- 本地模型路径是 `.ply` 文件（不是 `.pth`）
- 本地模型评估结果通常较差（只覆盖场景一小部分）
- 主要用于验证单个客户端训练是否成功

---

## 5. 辅助工具

### **5.1 检查模型健康状态**
```powershell
# 检查全局模型是否包含 NaN
python tools/check_model.py D:\githubdownloads\Fed3DGS_data\global-models\rubble\global_model.pth
```

### **5.2 修复包含 NaN 的模型**
```powershell
# 自动删除异常点并保存为 _fixed.pth
python tools/fix_model_nan.py D:\githubdownloads\Fed3DGS_data\global-models\rubble\global_model.pth
```

### **5.3 清理旧的 COLMAP 数据库**
```powershell
# 如果出现 "UNIQUE constraint failed" 错误
Remove-Item D:\githubdownloads\Fed3DGS_data\colmap-results\rubble\*\database.db -Force
```

### **5.4 检查已训练的客户端数量**
```powershell
# 统计本地模型数量
(Get-ChildItem D:\githubdownloads\Fed3DGS_data\local-models\rubble -Directory).Count

# 查看所有客户端编号
Get-ChildItem D:\githubdownloads\Fed3DGS_data\local-models\rubble -Directory | Select-Object Name
```

### **5.5 检查显存状态**
```powershell
nvidia-smi
```

---

## 6. 常见问题

### **Q1: COLMAP 命令找不到**
```
'colmap' 不是内部或外部命令
```

**解决：**
1. 确认 COLMAP 安装在 `D:\COLMAP\bin\colmap.exe`
2. 将 `D:\COLMAP\bin` 添加到系统 PATH 环境变量
3. 删除错误的用户级 PATH：`D:\githubdownloads\Fed3DGS_data\COLMAP\bin`
4. 重启所有终端

---

### **Q2: CUDA 非法内存访问错误**
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**诊断步骤：**
```powershell
# 1. 检查模型是否有 NaN
python tools/check_model.py <模型路径>

# 2. 如果有 NaN，原因是客户端数量太少
# 解决：训练更多客户端（至少 10 个）
```

---

### **Q3: SH degree 不匹配错误**
```
AssertionError: len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
或
size mismatch for 4.weight: copying a param with shape torch.Size([48, 64])...
```

**解决：** 确保三个地方的 `sh_degree` 一致：

1. **训练本地模型：** `gaussian-splatting/arguments/__init__.py` 第49行 → `self.sh_degree = 3`
2. **构建全局模型：** 添加 `--sh-degree 3`
3. **评估模型：** 添加 `--sh-degree 3`

---

### **Q4: 数据库重复键错误**
```
sqlite3.IntegrityError: UNIQUE constraint failed: images.name
```

**解决：**
```powershell
# 删除旧的数据库文件
Remove-Item D:\githubdownloads\Fed3DGS_data\colmap-results\rubble\*\database.db -Force
```

---

### **Q5: 全局模型全是 NaN**
```
scaling: NaN=True (100%)
opacity: NaN=True (85%)
```

**原因：** 客户端数量太少（2个不够）

**解决：**
1. 训练至少 10 个客户端
2. 使用更保守的学习率：
   ```
   --lr-opacity 0.01 --lr-mlp 1e-5
   ```

---

## 📊 **完整工作流程示例（10客户端）**

```powershell
# ============================================
# 步骤 1: 训练 10 个本地模型
# ============================================
scripts\client_training.bat 0 9 D:\githubdownloads\Fed3DGS_data\colmap-results\rubble D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble D:\githubdownloads\Fed3DGS_data\local-models\rubble

# 预计时间：5-10 小时

# ============================================
# 步骤 2: 检查训练结果
# ============================================
Get-ChildItem D:\githubdownloads\Fed3DGS_data\local-models\rubble -Directory

# 应该看到 10 个目录：00000, 00001, ..., 00009

# ============================================
# 步骤 3: 构建全局模型
# ============================================
python gaussian-splatting/build_global_model.py -w -o D:\githubdownloads\Fed3DGS_data\global-models\rubble-10clients -m D:\githubdownloads\Fed3DGS_data\local-models\rubble -i D:\githubdownloads\Fed3DGS_data\images\image-lists-rubble -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 --n-clients 10 --lr-opacity 0.01 --lr-mlp 1e-5

# 预计时间：10-30 分钟

# ============================================
# 步骤 4: 检查全局模型健康状态
# ============================================
python tools/check_model.py D:\githubdownloads\Fed3DGS_data\global-models\rubble-10clients\global_model.pth

# 应该看到：✅ 模型数据正常，没有 NaN 或 Inf 值

# ============================================
# 步骤 5: 评估全局模型
# ============================================
python gaussian-splatting/eval.py -w -o D:\githubdownloads\Fed3DGS_data\outputs\rubble-global -g D:\githubdownloads\Fed3DGS_data\global-models\rubble-10clients\global_model.pth -data D:\githubdownloads\Fed3DGS_data\pixsfm\rubble-pixsfm --sh-degree 3 -r 8

# 预计时间：5-10 分钟

# ============================================
# 步骤 6: 查看评估结果
# ============================================
notepad D:\githubdownloads\Fed3DGS_data\outputs\rubble-global\metrics.json

# 或
Get-Content D:\githubdownloads\Fed3DGS_data\outputs\rubble-global\metrics.json | ConvertFrom-Json
```

---

## 📈 **评估指标说明**

### **PSNR (Peak Signal-to-Noise Ratio)**
- **范围：** 0-∞ dB
- **越高越好**
- **参考标准：**
  - < 20 dB：较差
  - 20-25 dB：一般
  - 25-30 dB：良好 ⭐
  - > 30 dB：优秀

### **SSIM (Structural Similarity)**
- **范围：** 0-1
- **越高越好**
- **参考标准：**
  - < 0.7：较差
  - 0.7-0.85：一般
  - 0.85-0.95：良好 ⭐
  - > 0.95：优秀

### **LPIPS (Learned Perceptual Image Patch Similarity)**
- **范围：** 0-1
- **越低越好**（与 PSNR/SSIM 相反）
- **参考标准：**
  - < 0.1：优秀
  - 0.1-0.2：良好 ⭐
  - 0.2-0.4：一般
  - > 0.4：较差

---

## ⚙️ **系统要求**

- **GPU：** NVIDIA GPU with CUDA 支持（你的 RTX 4050 ✅）
- **显存：** 至少 6GB（你有 6GB ✅）
- **硬盘：** 至少 50GB 空闲空间
- **时间：** 
  - 10 客户端训练：5-10 小时
  - 全局模型构建：10-30 分钟
  - 评估：5-10 分钟

---

## 📝 **重要提醒**

### ⭐ **三个地方的 sh_degree 必须一致：**
1. 训练本地模型：`arguments/__init__.py` → `sh_degree = 3`
2. 构建全局模型：命令行参数 → `--sh-degree 3`
3. 评估模型：命令行参数 → `--sh-degree 3`

### ⭐ **客户端数量建议：**
- 测试/调试：3-5 个
- 小规模实验：10-20 个
- 论文级结果：50-200 个

### ⭐ **分辨率缩放建议：**
- RTX 4050 (6GB 显存)：建议 `-r 8` 或 `-r 16`
- 更大显存：可以用 `-r 4` 或 `-r 2`

---

## 🎓 **小多子的建议**

1. **先小规模测试**：训练 10 个客户端验证流程
2. **检查模型健康**：构建后用 `check_model.py` 检查
3. **保守的学习率**：使用 `--lr-opacity 0.01 --lr-mlp 1e-5`
4. **合理的分辨率**：评估时用 `-r 8`
5. **逐步扩展**：验证成功后再训练更多客户端

---

**祝你训练顺利！有问题随时找小多子！** 😊

