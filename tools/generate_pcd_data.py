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

def get_calibration_per_camera(calibration_file_path):
    """
    Reads the camera calibration file and returns the intrinsic and extrinsic parameters for each camera.
    """
    with open(calibration_file_path, 'r') as f:
        data = json.load(f)

    camera_params ={}
    for calib in data['sensors']:
        camera_name = calib['id']
        camera_names = camera_name.split('_')
        if len(camera_names) > 1:
            camera_name = camera_names[0] + '_' + f"{int(camera_names[1]):04d}"
        else:
            camera_name = camera_names[0] + '_0000'
        camera_intrinsic = np.array(calib['intrinsicMatrix']).reshape(3, 3)
        camera_intrinsic = camera_intrinsic[0,0], camera_intrinsic[1,1], camera_intrinsic[0,2], camera_intrinsic[1,2]
        camera_extrinsic = np.eye(4)
        camera_extrinsic[:3,:4] = np.array(calib['extrinsicMatrix']).reshape(3, 4)
        camera_params[camera_name] = {
            'intrinsic': camera_intrinsic,
            'extrinsic': camera_extrinsic
        }
    
    return camera_params

def get_point_cloud(rgb_image_per_camera, depth_image_per_camera, camera_params):
    pcd_combined = o3d.geometry.PointCloud()

    for camera_name in rgb_image_per_camera.keys():
        if depth_image_per_camera[camera_name] is None:
            continue

        rgb_image = rgb_image_per_camera[camera_name]
        H, W = rgb_image.shape[:2]
        depth_image = depth_image_per_camera[camera_name]
        depth_image = np.array(depth_image, dtype=np.float32) / 1000.0  # mm → meters

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_o3d = o3d.geometry.Image(rgb_image)
        depth_o3d = o3d.geometry.Image(depth_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)

        fx, fy, cx, cy = camera_params[camera_name]['intrinsic']
        cam_intr = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)

        # RGBD → PointCloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intr)

        extrinsic = camera_params[camera_name]['extrinsic']
        pcd.transform(np.linalg.inv(extrinsic)) 

        pcd_combined += pcd

    if len(pcd_combined.points) > 1000000:
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)

    return pcd_combined


data_root = 'dataset/MTMC_Tracking_2025'
out_dir = 'dataset/pcd_dataset'

split_set = ['train','val','test']

for split in split_set:
    if split == 'train' or split == 'val':
        num_skip_frame = 99
    else:
        num_skip_frame = 0
    domain_name_list = os.listdir(os.path.join(data_root, split))
    for domain_idx, domain_name in enumerate(domain_name_list):
        domain_path = os.path.join(data_root, split, domain_name)
        print(f'Processing {domain_path}...')
        camera_params = get_calibration_per_camera(os.path.join(domain_path, 'calibration.json'))
        camera_name_list = [name.split('.')[0] for name in os.listdir(os.path.join(domain_path,'videos'))]
        video_path_list = [os.path.join(domain_path, 'videos', f'{name}.mp4') for name in camera_name_list]
        depth_map_path_list = [os.path.join(domain_path, 'depth_maps', f'{name}.h5') for name in camera_name_list]

        new_camera_name_list = []
        for camera_name in camera_name_list:
            camera_names = camera_name.split('_')
            if len(camera_names) > 1:
                new_camera_name = camera_names[0] + '_' + f"{int(camera_names[1]):04d}"
            else:
                new_camera_name = camera_names[0] + '_0000'
            new_camera_name_list.append(new_camera_name)
        camera_name_list = new_camera_name_list

        depth_map_per_camera = {}
        for idx, depth_map_path in enumerate(depth_map_path_list):
            try:
                depth_map = h5py.File(depth_map_path, 'r')
                depth_map_per_camera[camera_name_list[idx]] = depth_map
            except Exception as e:
                print(f"Error opening depth map file {depth_map_path}: {e}")

        video_capture_per_camera = {}
        for idx, video_path in enumerate(video_path_list):
            try:
                video_capture = cv2.VideoCapture(video_path)
                if not video_capture.isOpened():
                    raise Exception(f"Cannot open video file: {video_path}")
                video_capture_per_camera[camera_name_list[idx]] = video_capture
            except Exception as e:
                print(f"Error opening video file {video_path}: {e}")
        
        frame_count = 0
        while True:
            rgb_image_per_camera = {}
            depth_image_per_camera = {}
            for camera_name in video_capture_per_camera:
                ret, frame = video_capture_per_camera[camera_name].read()
                if not ret:
                    break
                rgb_image_per_camera[camera_name] = frame
                
                try:
                    depth_map = depth_map_per_camera[camera_name][f'distance_to_image_plane_{frame_count:05d}.png'][:]
                except Exception as e:
                    print(f"Error reading depth map for {camera_name}: {e}")
                    depth_map = None
                depth_image_per_camera[camera_name] = depth_map

            output_path = f'{out_dir}/{split}/pcd/{domain_name}_{frame_count:05d}.ply'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            start_time = time.time()
            if not os.path.exists(output_path):
                with open(output_path, 'w') as f:
                    f.write(f"# {domain_name} frame {frame_count}\n")

                pcd = get_point_cloud(rgb_image_per_camera, depth_image_per_camera, camera_params)
                
                o3d.io.write_point_cloud(output_path, pcd)

            print(f"{output_path} Point cloud generated in {time.time() - start_time:.2f} seconds.")
            for camera_name in video_capture_per_camera.keys():
                for i in range(num_skip_frame):
                    ret, frame = video_capture_per_camera[camera_name].read()
            frame_count += num_skip_frame+1
            if frame_count >= 9000:
                for video in video_capture_per_camera.values():
                    video.release()
                break




# def process_pcd_slice(x,y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx):
#     pcd_path = os.path.join(output_dir, f"pcd/{file_name}_{x_idx:04d}_{y_idx:04d}.ply")
#     if os.path.exists(pcd_path):
#         print(f"File {pcd_path} already exists. Skipping...")
#         return
#     with open(pcd_path, 'w') as f:
#         f.write(f"# {file_name} slice at x={x}, y={y}\n")

#     voxel_min = np.array([x, y, min_bound[2]])
#     voxel_max = voxel_min + voxel_size
#     voxel_max = np.array([voxel_max[0], voxel_max[1], max_bound[2]])

#     # 4. 포인트 슬라이스
#     mask = np.all((points >= voxel_min) & (points < voxel_max), axis=1)
#     voxel_points = points[mask]
#     voxel_colors = colors[mask]

#     if len(voxel_points) > 0:
#         voxel_pcd = o3d.geometry.PointCloud()
#         voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
#         voxel_pcd.colors = o3d.utility.Vector3dVector(voxel_colors)

        

#         # 5. GT 슬라이스
#         gt_mask = (
#             (gt_df["x"] >= voxel_min[0]) & (gt_df["x"] < voxel_max[0]) &
#             (gt_df["y"] >= voxel_min[1]) & (gt_df["y"] < voxel_max[1]) &
#             (gt_df["z"] >= voxel_min[2]) & (gt_df["z"] < voxel_max[2])
#         )
#         gt_local = gt_df[gt_mask].copy()
#         os.remove(pcd_path)
#         if not gt_local.empty:
#             gt_path_out = os.path.join(output_dir, f"gt/{file_name}_{x_idx:04d}_{y_idx:04d}.txt")
#             gt_local.to_csv(gt_path_out, sep=' ', header=False, index=False)
#             o3d.io.write_point_cloud(pcd_path, voxel_pcd)

# split_set = ['train','val']

# for split in split_set:
#     input_pcd_root = f'/perception/dataset/PhysicalAI-SmartSpaces/pcd_dataset/{split}/pcd'
#     input_gt_root = f'/perception/dataset/PhysicalAI-SmartSpaces/pcd_dataset/{split}/gt'
#     output_dir = f"/perception/dataset/PhysicalAI-SmartSpaces/pcd_dataset_slice_20/{split}"
#     for file in os.listdir(input_pcd_root):
#         if 'Warehouse_014' not in file:
#             continue
#         if file.endswith('.ply'):
#             file_name = file.replace('.ply', '')
#             input_ply_path = os.path.join(input_pcd_root, file)
#             gt_path = os.path.join(input_gt_root, file.replace('.ply', '.txt'))
            
#             voxel_size = 20.0
#             os.makedirs(output_dir, exist_ok=True)

#             # 1. Load PCD
#             pcd = o3d.io.read_point_cloud(input_ply_path)
#             points = np.asarray(pcd.points)
#             colors = np.asarray(pcd.colors)

#             # 2. Load GT
#             gt_columns = ["label", "id", "x", "y", "z", "dx", "dy", "dz", "rx", "ry", "rz"]
#             gt_df = pd.read_csv(gt_path, sep=' ', header=None, names=gt_columns)

#             # 3. 전체 범위 계산
#             min_bound = points.min(axis=0)
#             max_bound = points.max(axis=0)

#             for x_idx, x in enumerate(np.arange(min_bound[0], max_bound[0], voxel_size/2)):
#                 for y_idx, y in enumerate(np.arange(min_bound[1], max_bound[1], voxel_size/2)):
#                     process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)
#                 y= max_bound[1] - voxel_size
#                 y_idx += 1
#                 # 마지막 슬라이스 처리
#                 process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)
#             x = max_bound[0] - voxel_size
#             x_idx += 1
#             for y_idx, y in enumerate(np.arange(min_bound[1], max_bound[1], voxel_size/2)):
#                 process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)
#             y = max_bound[1] - voxel_size
#             y_idx += 1
#             process_pcd_slice(x, y, voxel_size, points, colors, gt_df, min_bound, max_bound, file_name, output_dir,x_idx, y_idx)