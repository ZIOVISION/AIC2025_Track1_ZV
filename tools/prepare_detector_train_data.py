
import os
import cv2
import numpy as np
from PIL import Image
import h5py
import json
import subprocess
import open3d as o3d
import time
import pandas as pd

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate point clouds (PLY) from multi-camera RGB + depth sequences."
    )
    parser.add_argument("--pcd-data-root", type=str, default="dataset/pcd_dataset",
                        help="root directory of the PCD dataset.")
    parser.add_argument("--out-dir", type=str, default="dataset/pcd_dataset_slice_20",
                        help="Output root directory for generated PLY files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="Splits to process (space-separated). e.g., --splits train val")
    
    return parser.parse_args()

def process_pcd_slice(x,y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx):
    pcd_path = os.path.join(output_dir, f"pcd/{file_name}_{x_idx:04d}_{y_idx:04d}.ply")
    if os.path.exists(pcd_path):
        print(f"File {pcd_path} already exists. Skipping...")
        return
    with open(pcd_path, 'w') as f:
        f.write(f"# {file_name} slice at x={x}, y={y}\n")

    voxel_min = np.array([x, y, min_bound[2]])
    voxel_max = voxel_min + voxel_size
    voxel_max = np.array([voxel_max[0], voxel_max[1], max_bound[2]])

    # 4. Slice point cloud
    mask = np.all((points >= voxel_min) & (points < voxel_max), axis=1)
    voxel_points = points[mask]
    voxel_colors = colors[mask]

    if len(voxel_points) > 0:
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_pcd.colors = o3d.utility.Vector3dVector(voxel_colors)

        

        # 5. Slice ground truth
        gt_mask = (
            (gt_df["x"] >= voxel_min[0]) & (gt_df["x"] < voxel_max[0]) &
            (gt_df["y"] >= voxel_min[1]) & (gt_df["y"] < voxel_max[1]) &
            (gt_df["z"] >= voxel_min[2]) & (gt_df["z"] < voxel_max[2])
        )
        gt_local = gt_df[gt_mask].copy()
        os.remove(pcd_path)
        if not gt_local.empty:
            gt_path_out = os.path.join(output_dir, f"gt/{file_name}_{x_idx:04d}_{y_idx:04d}.txt")
            gt_local.to_csv(gt_path_out, sep=' ', header=False, index=False)
            o3d.io.write_point_cloud(pcd_path, voxel_pcd)


def main():
    args = parse_args()


    split_set = ['train','val']

    for split in split_set:
        input_pcd_root = f'{args.pcd_data_root}/{split}/pcd'
        input_gt_root = f'{args.pcd_data_root}/{split}/gt'
        output_dir = f"{args.out_dir}/{split}"


        for file in os.listdir(input_pcd_root):
            if file.endswith('.ply'):
                file_name = file.replace('.ply', '')
                _,_,frame_count = file_name.split('_')
                frame_count = int(frame_count)
                if frame_count % 100 != 0:
                    continue
                input_ply_path = os.path.join(input_pcd_root, file)
                gt_path = os.path.join(input_gt_root, file.replace('.ply', '.txt'))
                
                voxel_size = 20.0
                os.makedirs(output_dir, exist_ok=True)

                # 1. Load PCD
                pcd = o3d.io.read_point_cloud(input_ply_path)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                # 2. Load GT
                gt_columns = ["label", "id", "x", "y", "z", "dx", "dy", "dz", "rx", "ry", "rz"]
                gt_df = pd.read_csv(gt_path, sep=' ', header=None, names=gt_columns)

                # 3. Calculate overall bounds
                min_bound = points.min(axis=0)
                max_bound = points.max(axis=0)

                for x_idx, x in enumerate(np.arange(min_bound[0], max_bound[0], voxel_size/2)):
                    for y_idx, y in enumerate(np.arange(min_bound[1], max_bound[1], voxel_size/2)):
                        process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)
                    y= max_bound[1] - voxel_size
                    y_idx += 1
                    # Handle the last slice
                    process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)
                x = max_bound[0] - voxel_size
                x_idx += 1
                for y_idx, y in enumerate(np.arange(min_bound[1], max_bound[1], voxel_size/2)):
                    process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)
                y = max_bound[1] - voxel_size
                y_idx += 1
                process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)