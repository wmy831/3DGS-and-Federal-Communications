# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
"""
将COLMAP格式数据集转换为Fed3DGS格式

用法:
    python tools/convert_colmap_to_fed3dgs_1.py -i <COLMAP数据集路径> -o <输出路径>

示例:
    python tools/convert_colmap_to_fed3dgs_1.py -i "D:\githubdownloads\Fed3DGS_data\pixsfm\train" -o "D:\githubdownloads\Fed3DGS_data\pixsfm\train-converted"
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

# 添加路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gaussian-splatting'))
from scene.colmap_loader import (  # pyright: ignore[reportMissingImports]
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    qvec2rotmat
)

# 坐标系转换矩阵 (Mega-NeRF格式)
RDF_TO_DRB = torch.Tensor([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, -1]])


def w2c_to_c2w(R, T):
    """将world-to-camera转换为camera-to-world"""
    # R是旋转矩阵，T是平移向量
    # w2c = [R | T]
    # c2w = w2c^-1 = [R^T | -R^T @ T]
    R_inv = R.T
    T_inv = -R_inv @ T.reshape(3, 1)
    c2w = np.hstack([R_inv, T_inv])
    return c2w


def colmap_to_meganerf_c2w(c2w_colmap):
    """
    将COLMAP格式的c2w转换为Mega-NeRF格式
    
    这是create_db.py中转换的逆过程：
    create_db.py: Mega-NeRF c2w → COLMAP w2c
    这里: COLMAP c2w → Mega-NeRF c2w
    """
    c2w = torch.from_numpy(c2w_colmap).float()
    
    # 步骤1: 应用RDF_TO_DRB坐标系转换 (create_db.py第71行的逆变换)
    # create_db.py: RDF_TO_DRB.inverse() @ c2w[:3, :3] @ RDF_TO_DRB
    # 逆变换: RDF_TO_DRB @ c2w[:3, :3] @ RDF_TO_DRB.inverse()
    c2w_rot = RDF_TO_DRB @ c2w[:3, :3] @ RDF_TO_DRB.inverse()
    c2w_trans = RDF_TO_DRB @ c2w[:3, 3:]
    c2w_transformed = torch.cat([c2w_rot, c2w_trans], -1)
    
    # 步骤2: 应用列交换 (create_db.py第69行的逆变换)
    # create_db.py: torch.cat([-c2w[:, 1:2], c2w[:, :1], c2w[:, 2:]], 1)
    # 逆变换: torch.cat([c2w[:, 1:2], -c2w[:, 0:1], c2w[:, 2:]], 1)
    c2w_meganerf = torch.cat([c2w_transformed[:, 1:2], 
                              -c2w_transformed[:, 0:1], 
                              c2w_transformed[:, 2:]], 1)
    
    return c2w_meganerf.numpy()


def convert_colmap_to_fed3dgs(colmap_dir, output_dir):
    """
    将COLMAP格式数据集转换为Fed3DGS格式
    
    Args:
        colmap_dir: COLMAP数据集目录（包含images/和sparse/0/）
        output_dir: 输出目录（将创建train/metadata/和train/rgbs/）
    """
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)
    
    # 检查输入目录
    sparse_dir = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images"
    
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse目录不存在: {sparse_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    
    # 创建输出目录
    train_metadata_dir = output_dir / "train" / "metadata"
    train_rgbs_dir = output_dir / "train" / "rgbs"
    val_metadata_dir = output_dir / "val" / "metadata"
    val_rgbs_dir = output_dir / "val" / "rgbs"
    train_metadata_dir.mkdir(parents=True, exist_ok=True)
    train_rgbs_dir.mkdir(parents=True, exist_ok=True)
    val_metadata_dir.mkdir(parents=True, exist_ok=True)
    val_rgbs_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取COLMAP数据
    print("读取COLMAP相机参数...")
    cameras_extrinsic_file = sparse_dir / "images.bin"
    cameras_intrinsic_file = sparse_dir / "cameras.bin"
    
    try:
        cam_extrinsics = read_extrinsics_binary(str(cameras_extrinsic_file))
        cam_intrinsics = read_intrinsics_binary(str(cameras_intrinsic_file))
        print(f"  成功读取二进制格式")
    except:
        cameras_extrinsic_file = sparse_dir / "images.txt"
        cameras_intrinsic_file = sparse_dir / "cameras.txt"
        cam_extrinsics = read_extrinsics_text(str(cameras_extrinsic_file))
        cam_intrinsics = read_intrinsics_text(str(cameras_intrinsic_file))
        print(f"  成功读取文本格式")
    
    print(f"  找到 {len(cam_extrinsics)} 个相机")
    
    # 按图像名称排序
    sorted_images = sorted(cam_extrinsics.items(), key=lambda x: x[1].name)
    
    # 转换每个图像
    print("\n转换相机参数和复制图像...")
    print("  将每8张图像中的1张作为验证集（与readColmapSceneInfo的llffhold=8一致）")
    mappings = []
    
    # 收集所有c2w用于计算coordinates.pt
    all_c2ws = []
    
    for idx, (image_id, extr) in enumerate(tqdm(sorted_images, desc="处理图像")):
        # 获取相机内参
        intr = cam_intrinsics[extr.camera_id]
        
        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model == "PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            raise ValueError(f"不支持的相机模型: {intr.model}")
        
        # 获取图像尺寸
        width = intr.width
        height = intr.height
        
        # 将qvec/tvec转换为c2w
        R = np.transpose(qvec2rotmat(extr.qvec))  # world-to-camera旋转矩阵
        T = np.array(extr.tvec)  # world-to-camera平移向量
        
        # 转换为camera-to-world
        c2w_colmap = w2c_to_c2w(R, T)
        
        # 转换为Mega-NeRF格式的c2w
        c2w_meganerf = colmap_to_meganerf_c2w(c2w_colmap)
        
        # 收集c2w用于计算coordinates.pt
        all_c2ws.append(c2w_meganerf)
        
        # 创建metadata字典
        metadata = {
            'intrinsics': torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
            'c2w': torch.from_numpy(c2w_meganerf).float(),
            'H': height,
            'W': width
        }
        
        # 判断是训练集还是验证集（每8张取1张作为验证集，与readColmapSceneInfo的llffhold=8一致）
        is_val = (idx % 8 == 0)
        
        # 生成文件名（使用6位数字，从000000开始）
        image_name_base = Path(extr.name).stem
        new_index = f"{idx:06d}"
        
        # 选择输出目录
        if is_val:
            metadata_file = val_metadata_dir / f"{new_index}.pt"
            dest_image_dir = val_rgbs_dir
        else:
            metadata_file = train_metadata_dir / f"{new_index}.pt"
            dest_image_dir = train_rgbs_dir
        
        # 保存metadata
        torch.save(metadata, metadata_file)
        
        # 复制图像
        source_image = images_dir / extr.name
        if not source_image.exists():
            # 尝试不同的扩展名
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                alt_source = images_dir / f"{Path(extr.name).stem}{ext}"
                if alt_source.exists():
                    source_image = alt_source
                    break
        
        if source_image.exists():
            dest_image = dest_image_dir / f"{new_index}.jpg"
            # 如果是jpg，直接复制；否则转换
            if source_image.suffix.lower() in ['.jpg', '.jpeg']:
                shutil.copy2(source_image, dest_image)
            else:
                img = Image.open(source_image).convert('RGB')
                img.save(dest_image, 'JPEG', quality=95)
            
            # 记录映射
            mappings.append(f"{extr.name},{new_index}.pt")
        else:
            print(f"\n警告: 图像文件不存在: {source_image}")
    
    # 保存mappings.txt
    mappings_file = output_dir / "mappings.txt"
    with open(mappings_file, 'w') as f:
        for mapping in mappings:
            f.write(mapping + '\n')
    
    # 计算并保存coordinates.pt（坐标归一化参数，虽然代码中未使用，但为完整性生成）
    print("\n计算坐标归一化参数...")
    all_c2ws_tensor = torch.from_numpy(np.stack(all_c2ws)).float()
    cam_centers = all_c2ws_tensor[:, :3, 3]  # 提取相机中心位置
    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)
    dists = torch.norm(cam_centers - avg_cam_center, dim=1, keepdim=True)
    diagonal = torch.max(dists).item()
    radius = diagonal * 1.1
    translate = -avg_cam_center.squeeze().numpy()
    
    coordinates = {
        'translate': translate,
        'radius': radius
    }
    coordinates_file = output_dir / "coordinates.pt"
    torch.save(coordinates, coordinates_file)
    
    # 统计信息
    n_train = len([f for f in train_metadata_dir.glob("*.pt")])
    n_val = len([f for f in val_metadata_dir.glob("*.pt")])
    
    print(f"\n✅ 转换完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   总图像数: {len(sorted_images)}")
    print(f"   训练集: {n_train} 张图像")
    print(f"   验证集: {n_val} 张图像")
    print(f"   Train Metadata: {train_metadata_dir}")
    print(f"   Train Images: {train_rgbs_dir}")
    print(f"   Val Metadata: {val_metadata_dir}")
    print(f"   Val Images: {val_rgbs_dir}")
    print(f"   映射文件: {mappings_file}")
    print(f"   坐标归一化: {coordinates_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将COLMAP格式数据集转换为Fed3DGS格式')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='COLMAP数据集目录（包含images/和sparse/0/）')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='输出目录（将创建train/metadata/和train/rgbs/）')
    
    args = parser.parse_args()
    
    convert_colmap_to_fed3dgs(args.input, args.output)

