# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
"""
从COLMAP格式数据集生成客户端数据列表（方案3的修改版本）

用法:
    python tools/gen_client_data_colmap_1.py -d <COLMAP数据集目录> -o <输出目录> --n-clients <客户端数量>

示例:
    python tools/gen_client_data_colmap_1.py -d "D:\githubdownloads\Fed3DGS_data\pixsfm\train" -o "D:\githubdownloads\Fed3DGS_data\image-lists\train-colmap" --n-clients 20
"""

import os
import random
import sys
import numpy as np
import torch
from pathlib import Path
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
    R_inv = R.T
    T_inv = -R_inv @ T.reshape(3, 1)
    c2w = np.hstack([R_inv, T_inv])
    return c2w


def colmap_to_meganerf_c2w(c2w_colmap):
    """将COLMAP格式的c2w转换为Mega-NeRF格式"""
    c2w = torch.from_numpy(c2w_colmap).float()
    
    # 步骤1: 应用RDF_TO_DRB坐标系转换
    c2w_rot = RDF_TO_DRB @ c2w[:3, :3] @ RDF_TO_DRB.inverse()
    c2w_trans = RDF_TO_DRB @ c2w[:3, 3:]
    c2w_transformed = torch.cat([c2w_rot, c2w_trans], -1)
    
    # 步骤2: 应用列交换
    c2w_meganerf = torch.cat([c2w_transformed[:, 1:2], 
                              -c2w_transformed[:, 0:1], 
                              c2w_transformed[:, 2:]], 1)
    
    return c2w_meganerf.numpy()


def gen_client_data(c2ws: np.ndarray, n_data: int):
    """
    Args:
        c2ws (np.ndarray): camera extrinsic (camera2world) that is
                        an ndarray of shape (#cameras, 3, 4).
                        the coordinate system is following mega-nerf,
                        i.e., (down, right, backward)
        n_data (int): a number of data for a client
                        
    Returns:
        indices (np.ndarray): client's data indices
    """
    n_cameras = c2ws.shape[0]
    xyz_coord = c2ws[:, :3, -1]  # (#cameras, 3)
    
    base_camera_idx = np.random.randint(0, n_cameras)
    center_xyz = xyz_coord[base_camera_idx]
    dists = np.sum(np.square(xyz_coord - center_xyz), -1)
    
    indices = np.argsort(dists, 0)[:n_data]
    return np.sort(indices)


def load_colmap_c2ws(colmap_dir):
    """
    从COLMAP格式数据集加载所有相机的c2w矩阵
    
    Args:
        colmap_dir: COLMAP数据集目录（包含sparse/0/和images/）
    
    Returns:
        c2ws: numpy array of shape (n_cameras, 3, 4)
        image_names: list of image filenames
    """
    colmap_dir = Path(colmap_dir)
    sparse_dir = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images"
    
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse目录不存在: {sparse_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    
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
    
    # 转换每个相机的c2w
    print("转换相机参数...")
    c2ws = []
    image_names = []
    
    for image_id, extr in tqdm(sorted_images, desc="处理相机"):
        # 将qvec/tvec转换为c2w
        R = np.transpose(qvec2rotmat(extr.qvec))  # world-to-camera旋转矩阵
        T = np.array(extr.tvec)  # world-to-camera平移向量
        
        # 转换为camera-to-world
        c2w_colmap = w2c_to_c2w(R, T)
        
        # 转换为Mega-NeRF格式的c2w
        c2w_meganerf = colmap_to_meganerf_c2w(c2w_colmap)
        
        c2ws.append(c2w_meganerf)
        image_names.append(extr.name)
    
    c2ws = np.stack(c2ws)
    return c2ws, image_names


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='从COLMAP格式数据集生成客户端数据列表')
    parser.add_argument('--dataset-dir', '-d',
                        required=True,
                        type=str,
                        help='COLMAP数据集目录（包含sparse/0/和images/）')
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        type=str,
                        help='输出目录')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random seed')
    parser.add_argument('--n-clients',
                        default=200,
                        type=int,
                        help='number of clients')
    parser.add_argument('--n-data-min', '-min',
                        default=100,
                        type=int,
                        help='minimum number of clients data')
    parser.add_argument('--n-data-max', '-max',
                        default=200,
                        type=int,
                        help='maximum number of clients data')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 从COLMAP格式加载c2w
    c2ws, image_names = load_colmap_c2ws(args.dataset_dir)
    
    print('split data')
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.n_clients):
        n_data = np.random.randint(args.n_data_min, args.n_data_max + 1)
        indices = gen_client_data(c2ws, n_data)
        training_image_names = [image_names[idx] for idx in indices]
        np.savetxt(os.path.join(args.output_dir, str(i).zfill(5) + '.txt'), training_image_names, fmt="%s")
    
    print(f"\n✅ 完成！生成了 {args.n_clients} 个客户端的图像列表")
    print(f"   输出目录: {args.output_dir}")

