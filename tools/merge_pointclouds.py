# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
"""
合并多个点云.bin文件为一个点云文件

功能：
1. 读取多个点云.bin文件
2. 合并点云（处理重复的点ID）
3. 保存为单个.bin文件

用法:
    python tools/merge_pointclouds.py -i <点云文件列表或目录> -o <输出文件>
    
示例:
    # 从文件列表合并
    python tools/merge_pointclouds.py -i "000001.bin,000005.bin,000012.bin" -o "merged.bin"
    
    # 从目录合并所有.bin文件
    python tools/merge_pointclouds.py -i "pointclouds_dir/" -o "merged.bin"
    
    # 合并客户端点云文件夹
    python tools/merge_pointclouds.py -i "00000_pointclouds/" -o "00000_merged.bin"
"""
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# 导入点云读写函数
sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import (
    read_points3D_binary, read_points3D_text,
    write_points3D_binary, write_points3D_text,
    Point3D
)


def merge_pointclouds(pointcloud_files, output_file, merge_strategy='union'):
    """
    合并多个点云文件
    
    Args:
        pointcloud_files: 点云文件路径列表
        output_file: 输出文件路径
        merge_strategy: 合并策略
            - 'union': 合并所有点，如果点ID重复则合并观测信息
            - 'replace': 如果点ID重复，用后面的文件覆盖前面的
    
    Returns:
        合并后的点云字典
    """
    merged_points = {}
    total_points = 0
    
    print(f"开始合并 {len(pointcloud_files)} 个点云文件...")
    
    for idx, pc_file in enumerate(pointcloud_files):
        pc_path = Path(pc_file)
        if not pc_path.exists():
            print(f"  警告: 文件不存在，跳过: {pc_file}")
            continue
        
        try:
            # 读取点云文件
            if pc_path.suffix == '.bin':
                points3D = read_points3D_binary(str(pc_path))
            elif pc_path.suffix == '.txt':
                points3D = read_points3D_text(str(pc_path))
            else:
                print(f"  警告: 不支持的文件格式，跳过: {pc_file}")
                continue
            
            print(f"  [{idx+1}/{len(pointcloud_files)}] 读取 {pc_path.name}: {len(points3D)} 个点")
            total_points += len(points3D)
            
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
                            # 使用字典来去重，保持顺序
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
                                xyz=point.xyz,  # 使用新点的位置（或可以取平均）
                                rgb=point.rgb,  # 使用新点的颜色（或可以取平均）
                                error=error,
                                image_ids=unique_image_ids,
                                point2D_idxs=unique_point2D_idxs
                            )
                        else:
                            # 不同位置的点，但ID相同（可能是不同重建中的点）
                            # 使用新点覆盖（或可以创建新ID）
                            print(f"    警告: 点ID {point_id} 在不同文件中有不同位置，使用新点覆盖")
                            merged_points[point_id] = point
                    else:  # replace策略
                        # 直接覆盖
                        merged_points[point_id] = point
                else:
                    # 新点，直接添加
                    merged_points[point_id] = point
                    
        except Exception as e:
            print(f"  错误: 读取文件失败 {pc_file}: {e}")
            continue
    
    print(f"\n合并完成:")
    print(f"  总输入点数: {total_points}")
    print(f"  合并后点数: {len(merged_points)}")
    print(f"  去重减少: {total_points - len(merged_points)} 个点")
    
    # 保存合并后的点云
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if output_path.suffix == '.bin':
            write_points3D_binary(merged_points, str(output_path))
            print(f"  已保存二进制格式: {output_path}")
        elif output_path.suffix == '.txt':
            write_points3D_text(merged_points, str(output_path))
            print(f"  已保存文本格式: {output_path}")
        else:
            # 默认保存为.bin格式
            output_path = output_path.with_suffix('.bin')
            write_points3D_binary(merged_points, str(output_path))
            print(f"  已保存二进制格式: {output_path}")
    except Exception as e:
        print(f"  错误: 保存文件失败: {e}")
        raise
    
    return merged_points


def collect_pointcloud_files(input_path):
    """
    收集点云文件路径
    
    Args:
        input_path: 输入路径（可以是文件列表、目录或单个文件）
    
    Returns:
        点云文件路径列表
    """
    input_path = Path(input_path)
    pointcloud_files = []
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix in ['.bin', '.txt']:
            pointcloud_files = [input_path]
        else:
            # 可能是包含文件列表的文本文件
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pc_file = Path(line)
                        if pc_file.exists() and pc_file.suffix in ['.bin', '.txt']:
                            pointcloud_files.append(pc_file)
    elif input_path.is_dir():
        # 目录：查找所有.bin和.txt文件
        pointcloud_files = sorted(list(input_path.glob("*.bin")) + list(input_path.glob("*.txt")))
    else:
        # 可能是逗号分隔的文件列表字符串
        files_str = str(input_path)
        if ',' in files_str:
            for file_str in files_str.split(','):
                file_str = file_str.strip()
                pc_file = Path(file_str)
                if pc_file.exists() and pc_file.suffix in ['.bin', '.txt']:
                    pointcloud_files.append(pc_file)
        else:
            raise ValueError(f"无法解析输入路径: {input_path}")
    
    if len(pointcloud_files) == 0:
        raise ValueError(f"未找到任何点云文件: {input_path}")
    
    return pointcloud_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='合并多个点云.bin文件为一个点云文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 从文件列表合并
    python tools/merge_pointclouds.py -i "000001.bin,000005.bin,000012.bin" -o "merged.bin"
    
    # 从目录合并所有.bin文件
    python tools/merge_pointclouds.py -i "pointclouds_dir/" -o "merged.bin"
    
    # 合并客户端点云文件夹
    python tools/merge_pointclouds.py -i "00000_pointclouds/" -o "00000_merged.bin"
        """
    )
    parser.add_argument('-i', '--input',
                        required=True,
                        type=str,
                        help='输入：点云文件路径（可以是逗号分隔的文件列表、目录路径或单个文件）')
    parser.add_argument('-o', '--output',
                        required=True,
                        type=str,
                        help='输出点云文件路径（.bin或.txt）')
    parser.add_argument('--merge-strategy',
                        choices=['union', 'replace'],
                        default='union',
                        help='合并策略：union=合并观测信息, replace=覆盖（默认: union）')
    parser.add_argument('--format',
                        choices=['bin', 'txt', 'both'],
                        default='bin',
                        help='输出格式：bin=二进制, txt=文本, both=两种格式（默认: bin）')
    
    args = parser.parse_args()
    
    # 收集点云文件
    try:
        pointcloud_files = collect_pointcloud_files(args.input)
        print(f"找到 {len(pointcloud_files)} 个点云文件:")
        for pc_file in pointcloud_files:
            print(f"  - {pc_file}")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 合并点云
    try:
        merged_points = merge_pointclouds(
            pointcloud_files,
            args.output,
            merge_strategy=args.merge_strategy
        )
        
        # 如果需要同时保存文本格式
        if args.format in ['txt', 'both']:
            output_path = Path(args.output)
            txt_output = output_path.with_suffix('.txt')
            write_points3D_text(merged_points, str(txt_output))
            print(f"  已保存文本格式: {txt_output}")
        
        print(f"\n✅ 合并完成！")
        print(f"   输出文件: {args.output}")
        print(f"   合并点数: {len(merged_points)}")
        
    except Exception as e:
        print(f"错误: 合并失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

