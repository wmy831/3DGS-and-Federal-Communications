# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
#
# 修改版本：输出结构与 client_training.sh:10 对齐
# 输出目录结构：{client_id}/sparse/0/{points3D.bin, images.bin, cameras.bin}

import os
import sys
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from tqdm import tqdm

# 导入点云读写函数
sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import (
    read_points3D_binary, read_points3D_text,
    write_points3D_binary, write_points3D_text,
    read_images_binary, read_images_text,
    read_cameras_binary, read_cameras_text,
    write_images_binary, write_cameras_binary,
    Point3D, Image, Camera
)


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
    xyz_coord = c2ws[:, :3, -1] # (#cameras, 3)
    
    base_camera_idx = np.random.randint(0, n_cameras)
    center_xyz = xyz_coord[base_camera_idx]
    dists = np.sum(np.square(xyz_coord - center_xyz), -1)
    
    indices = np.argsort(dists, 0)[:n_data]
    return np.sort(indices)


def get_pointcloud_bin_path(dataset_dir, image_index):
    """
    获取指定图像的点云.bin文件路径
    
    Args:
        dataset_dir: 数据集根目录
        image_index: 图像索引（如 "000001"）
    
    Returns:
        bin_file_path: 点云.bin文件路径（如果存在），否则返回None
    """
    pointcloud_dir = Path(dataset_dir) / "train" / "pointclouds" / image_index
    bin_file = pointcloud_dir / "points3D.bin"
    
    if bin_file.exists():
        return bin_file
    return None


def merge_pointclouds_for_client(pointcloud_files, merge_strategy='union'):
    """
    合并多个点云文件为一个点云字典
    
    Args:
        pointcloud_files: 点云文件路径列表
        merge_strategy: 合并策略
            - 'union': 合并所有点，如果点ID重复则合并观测信息
            - 'replace': 如果点ID重复，用后面的文件覆盖前面的
    
    Returns:
        合并后的点云字典
    """
    merged_points = {}
    
    for pc_file in pointcloud_files:
        pc_path = Path(pc_file)
        if not pc_path.exists():
            continue
        
        try:
            # 读取点云文件
            if pc_path.suffix == '.bin':
                points3D = read_points3D_binary(str(pc_path))
            elif pc_path.suffix == '.txt':
                points3D = read_points3D_text(str(pc_path))
            else:
                continue
            
            # 合并点云
            for point_id, point in points3D.items():
                if point_id in merged_points:
                    # 点ID已存在，需要合并观测信息
                    existing_point = merged_points[point_id]
                    
                    if merge_strategy == 'union':
                        # 合并策略：合并image_ids和point2D_idxs
                        # 检查点是否相同（位置、颜色、误差）
                        if np.allclose(existing_point.xyz, point.xyz, atol=1e-6):
                            # 相同位置的点，合并观测信息
                            # 合并image_ids和point2D_idxs，去重
                            combined_image_ids = np.concatenate([existing_point.image_ids, point.image_ids])
                            combined_point2D_idxs = np.concatenate([existing_point.point2D_idxs, point.point2D_idxs])
                            
                            # 去重：保留唯一的(image_id, point2D_idx)对
                            unique_pairs = {}
                            for img_id, pt2d_idx in zip(combined_image_ids, combined_point2D_idxs):
                                key = (int(img_id), int(pt2d_idx))
                                if key not in unique_pairs:
                                    unique_pairs[key] = (img_id, pt2d_idx)
                            
                            # 重建数组
                            unique_image_ids = np.array([p[0] for p in unique_pairs.values()], dtype=np.int32)
                            unique_point2D_idxs = np.array([p[1] for p in unique_pairs.values()], dtype=np.int32)
                            
                            # 使用较小的误差值（更精确的点）
                            error = min(existing_point.error, point.error)
                            
                            merged_points[point_id] = Point3D(
                                id=point_id,
                                xyz=point.xyz,
                                rgb=point.rgb,
                                error=error,
                                image_ids=unique_image_ids,
                                point2D_idxs=unique_point2D_idxs
                            )
                        else:
                            # 不同位置的点，但ID相同，使用新点覆盖
                            merged_points[point_id] = point
                    else:  # replace策略
                        # 直接覆盖
                        merged_points[point_id] = point
                else:
                    # 新点，直接添加
                    merged_points[point_id] = point
                    
        except Exception as e:
            print(f"  警告: 读取点云文件失败 {pc_file}: {e}")
            continue
    
    return merged_points


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='生成客户端COLMAP文件（合并分配图像的点云，提取images.bin和cameras.bin）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python tools/gen_client_data_with_pointclouds_modified.py \\
        -d "D:\\githubdownloads\\Fed3DGS_data\\pixsfm\\truck-fed3dgs" \\
        -o "D:\\githubdownloads\\Fed3DGS_data\\colmap-results\\truck" \\
        -c "D:\\githubdownloads\\Fed3DGS_data\\pixsfm\\truck" \\
        --n-clients 200

输出结构:
    00000/sparse/0/
        ├── points3D.bin  # 合并后的点云（image_ids已更新）
        ├── images.bin    # 过滤后的图像外参（ID从1开始）
        └── cameras.bin   # 过滤后的相机内参（ID从1开始）
    00001/sparse/0/
        ├── points3D.bin
        ├── images.bin
        └── cameras.bin
    ...
        """
    )
    parser.add_argument('--dataset-dir', '-d',
                        required=True,
                        type=str,
                        help='数据集根目录（包含train/rgbs/和train/pointclouds/）')
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        type=str,
                        help='输出目录（将生成{client_id}/sparse/0/目录结构）')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='随机种子（默认: 1）')
    parser.add_argument('--n-clients',
                        default=200,
                        type=int,
                        help='客户端数量（默认: 200）')
    parser.add_argument('--n-data-min', '-min',
                        default=100,
                        type=int,
                        help='每个客户端最少数据量（默认: 100）')
    parser.add_argument('--n-data-max', '-max',
                        default=200,
                        type=int,
                        help='每个客户端最多数据量（默认: 200）')
    parser.add_argument('--warn-missing-pointclouds',
                        action='store_true',
                        help='当点云文件不存在时发出警告')
    parser.add_argument('--merge-strategy',
                        choices=['union', 'replace'],
                        default='union',
                        help='点云合并策略：union=合并观测信息, replace=覆盖（默认: union）')
    parser.add_argument('--colmap-dir', '-c',
                        required=True,
                        type=str,
                        help='原始COLMAP数据集目录（包含sparse/0/images.bin和cameras.bin）')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    colmap_dir = Path(args.colmap_dir)
    
    # 检查数据集目录
    train_rgbs_dir = dataset_dir / "train" / "rgbs"
    train_metadata_dir = dataset_dir / "train" / "metadata"
    train_pointclouds_dir = dataset_dir / "train" / "pointclouds"
    
    if not train_rgbs_dir.exists():
        raise FileNotFoundError(f"训练集RGB目录不存在: {train_rgbs_dir}")
    if not train_metadata_dir.exists():
        raise FileNotFoundError(f"训练集metadata目录不存在: {train_metadata_dir}")
    
    # 检查COLMAP目录
    colmap_sparse_dir = colmap_dir / "sparse" / "0"
    colmap_images_file = colmap_sparse_dir / "images.bin"
    colmap_cameras_file = colmap_sparse_dir / "cameras.bin"
    
    if not colmap_images_file.exists():
        # 尝试文本格式
        colmap_images_file = colmap_sparse_dir / "images.txt"
        colmap_cameras_file = colmap_sparse_dir / "cameras.txt"
        if not colmap_images_file.exists():
            raise FileNotFoundError(f"COLMAP images文件不存在: {colmap_sparse_dir / 'images.bin'} 或 {colmap_sparse_dir / 'images.txt'}")
    
    # 检查点云目录（可选）
    has_pointclouds = train_pointclouds_dir.exists()
    if not has_pointclouds:
        print(f"警告: 点云目录不存在: {train_pointclouds_dir}")
        print("  将只分配图像，不分配点云文件")
    
    # 读取原始COLMAP数据
    print('读取原始COLMAP数据...')
    try:
        if colmap_images_file.suffix == '.bin':
            colmap_images = read_images_binary(str(colmap_images_file))
            colmap_cameras = read_cameras_binary(str(colmap_cameras_file))
        else:
            colmap_images = read_images_text(str(colmap_images_file))
            colmap_cameras = read_cameras_text(str(colmap_cameras_file))
        print(f'  成功读取 {len(colmap_images)} 个图像和 {len(colmap_cameras)} 个相机')
    except Exception as e:
        raise RuntimeError(f"读取COLMAP数据失败: {e}")
    
    # 建立图像文件名到COLMAP image_id的映射
    print('建立图像文件名映射...')
    image_name_to_colmap_id = {}
    for image_id, image in colmap_images.items():
        image_name = image.name
        # 只保留文件名（去除路径）
        image_name_base = os.path.basename(image_name)
        image_name_to_colmap_id[image_name_base] = image_id
    print(f'  建立了 {len(image_name_to_colmap_id)} 个图像映射')
    
    # 读取图像文件名
    fnames = sorted(os.listdir(train_rgbs_dir))
    print(f'找到 {len(fnames)} 张图像')
    
    # 加载相机参数
    print('加载相机参数...')
    c2ws = []
    valid_indices = []  # 记录有效的图像索引
    for idx, fname in enumerate(tqdm(fnames, desc="读取metadata")):
        try:
            metadata_file = train_metadata_dir / f"{Path(fname).stem}.pt"
            if metadata_file.exists():
                c2w = torch.load(str(metadata_file))['c2w'].numpy()
                c2ws.append(c2w)
                valid_indices.append(idx)
            else:
                print(f"  警告: metadata文件不存在: {metadata_file}")
        except Exception as e:
            print(f"  警告: 读取 {fname} 的metadata失败: {e}")
    
    if len(c2ws) == 0:
        raise ValueError("无法加载任何相机参数！")
    
    c2ws = np.stack(c2ws)
    print(f'成功加载 {len(c2ws)} 个相机参数')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    total_with_pointclouds = 0
    total_without_pointclouds = 0
    
    # 为每个客户端分配数据
    print(f'\n开始分配数据到 {args.n_clients} 个客户端...')
    for i in tqdm(range(args.n_clients), desc="分配客户端数据"):
        # 生成客户端数据索引
        n_data = np.random.randint(args.n_data_min, args.n_data_max + 1)
        # 注意：这里使用有效索引的范围
        if len(valid_indices) < n_data:
            print(f"  警告: 客户端 {i:05d} 需要 {n_data} 个数据，但只有 {len(valid_indices)} 个有效数据")
            n_data = len(valid_indices)
        
        # 从有效索引中生成数据分配
        # 需要将c2ws的索引映射回原始fnames的索引
        if n_data > 0:
            # 在有效索引范围内生成分配
            valid_c2ws_indices = gen_client_data(c2ws, min(n_data, len(c2ws)))
            # 映射回原始图像索引
            selected_indices = [valid_indices[idx] for idx in valid_c2ws_indices]
            selected_fnames = [fnames[idx] for idx in selected_indices]
        else:
            selected_fnames = []
        
        client_id = str(i).zfill(5)
        
        # 创建客户端输出目录
        client_output_dir = output_dir / client_id / "sparse" / "0"
        client_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        image_list = []
        pointcloud_bin_files = []  # 存储点云.bin文件路径和对应的图像索引
        selected_colmap_image_ids = []  # 存储选中的COLMAP image_id
        
        # 处理每个图像
        for fname in selected_fnames:
            image_index = Path(fname).stem  # 例如 "000001"
            image_list.append(fname)
            
            # 查找COLMAP中的image_id
            if fname in image_name_to_colmap_id:
                colmap_image_id = image_name_to_colmap_id[fname]
                selected_colmap_image_ids.append(colmap_image_id)
            else:
                if args.warn_missing_pointclouds:
                    print(f"  警告: 客户端 {client_id} 的图像 {fname} 在COLMAP中未找到")
                continue
            
            # 检查点云.bin文件是否存在
            bin_file_path = None
            if has_pointclouds:
                bin_file_path = get_pointcloud_bin_path(dataset_dir, image_index)
            
            if bin_file_path:
                total_with_pointclouds += 1
                pointcloud_bin_files.append((image_index, bin_file_path, colmap_image_id))
            else:
                total_without_pointclouds += 1
                if args.warn_missing_pointclouds:
                    print(f"  警告: 客户端 {client_id} 的图像 {fname} 没有对应的点云.bin文件")
        
        # 如果没有选中的图像，跳过
        if len(selected_colmap_image_ids) == 0:
            continue
        
        # 过滤并重新分配COLMAP图像和相机
        # 1. 收集需要的图像
        client_images = {}
        client_camera_ids = set()
        for colmap_image_id in selected_colmap_image_ids:
            if colmap_image_id in colmap_images:
                image = colmap_images[colmap_image_id]
                client_images[colmap_image_id] = image
                client_camera_ids.add(image.camera_id)
        
        # 2. 收集需要的相机
        client_cameras = {}
        for camera_id in client_camera_ids:
            if camera_id in colmap_cameras:
                client_cameras[camera_id] = colmap_cameras[camera_id]
        
        # 3. 重新分配ID（从1开始连续）
        # 相机ID映射
        camera_id_mapping = {}
        new_camera_id = 1
        for old_camera_id in sorted(client_cameras.keys()):
            camera_id_mapping[old_camera_id] = new_camera_id
            new_camera_id += 1
        
        # 图像ID映射
        image_id_mapping = {}
        new_image_id = 1
        for old_image_id in sorted(client_images.keys()):
            image_id_mapping[old_image_id] = new_image_id
            new_image_id += 1
        
        # 4. 创建新的图像和相机字典（使用新ID）
        new_client_images = {}
        for old_image_id, image in client_images.items():
            new_image_id = image_id_mapping[old_image_id]
            new_camera_id = camera_id_mapping[image.camera_id]
            new_client_images[new_image_id] = Image(
                id=new_image_id,
                qvec=image.qvec,
                tvec=image.tvec,
                camera_id=new_camera_id,
                name=image.name,
                xys=image.xys,
                point3D_ids=image.point3D_ids
            )
        
        new_client_cameras = {}
        for old_camera_id, camera in client_cameras.items():
            new_camera_id = camera_id_mapping[old_camera_id]
            new_client_cameras[new_camera_id] = Camera(
                id=new_camera_id,
                model=camera.model,
                width=camera.width,
                height=camera.height,
                params=camera.params
            )
        
        # 5. 合并点云并更新image_ids
        merged_points = {}
        if pointcloud_bin_files:
            # 收集所有点云文件路径
            bin_file_paths = [bin_path for _, bin_path, _ in pointcloud_bin_files]
            
            # 合并点云
            merged_points = merge_pointclouds_for_client(
                bin_file_paths,
                merge_strategy=args.merge_strategy
            )
            
            # 更新点云中的image_ids以匹配新的image_id
            updated_points = {}
            for point_id, point in merged_points.items():
                # 更新image_ids：将旧的COLMAP image_id映射到新的image_id
                new_image_ids = []
                new_point2D_idxs = []
                for img_id, pt2d_idx in zip(point.image_ids, point.point2D_idxs):
                    old_img_id = int(img_id)
                    if old_img_id in image_id_mapping:
                        new_img_id = image_id_mapping[old_img_id]
                        new_image_ids.append(new_img_id)
                        new_point2D_idxs.append(pt2d_idx)
                
                if len(new_image_ids) > 0:
                    updated_points[point_id] = Point3D(
                        id=point.id,
                        xyz=point.xyz,
                        rgb=point.rgb,
                        error=point.error,
                        image_ids=np.array(new_image_ids, dtype=np.int32),
                        point2D_idxs=np.array(new_point2D_idxs, dtype=np.int32)
                    )
            merged_points = updated_points
        
        # 6. 保存文件到 {client_id}/sparse/0/
        if len(new_client_images) > 0 and len(new_client_cameras) > 0:
            try:
                # 保存images.bin
                images_output_file = client_output_dir / "images.bin"
                write_images_binary(new_client_images, str(images_output_file))
                
                # 保存cameras.bin
                cameras_output_file = client_output_dir / "cameras.bin"
                write_cameras_binary(new_client_cameras, str(cameras_output_file))
                
                # 保存points3D.bin
                if len(merged_points) > 0:
                    points3d_output_file = client_output_dir / "points3D.bin"
                    write_points3D_binary(merged_points, str(points3d_output_file))
                
            except Exception as e:
                print(f"  警告: 保存客户端 {client_id} 的COLMAP文件失败: {e}")
    
    # 打印统计信息
    print(f'\n 数据分配完成！')
    print(f'   输出目录: {output_dir}')
    print(f'   客户端数量: {args.n_clients}')
    print(f'   总图像数: {len(fnames)}')
    print(f'   有效相机参数: {len(c2ws)}')
    print(f'   有点云的图像: {total_with_pointclouds}')
    print(f'   无点云的图像: {total_without_pointclouds}')

