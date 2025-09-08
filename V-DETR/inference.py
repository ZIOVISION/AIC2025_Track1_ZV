import argparse
import numpy as np
import torch
import math
import open3d as o3d
from plyfile import PlyData

from torch.utils.data import DataLoader, SequentialSampler

from datasets import build_dataset
from engine import evaluate
from models import build_model
from utils.dist import batch_dict_to_cuda
from utils.box_util import apply_nms_to_detections

import yaml
from types import SimpleNamespace
np.set_printoptions(precision=4, suppress=True)

import tqdm
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])  # Color mean for normalization
import os
import time

def load_config():
    parser = argparse.ArgumentParser(description="V-DETR config loader")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Path to the model checkpoint for testing')
    parser.add_argument('--output_dir', type=str, default='/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025_outputs/detection', help='Directory to save outputs')
    parser.add_argument('--data_root', type=str, default='/perception/dataset/PhysicalAI-SmartSpaces/pcd_dataset', help='Directory containing the dataset')
    parser.add_argument('--split_name', type=str, default='test', help='Dataset split name (e.g., test)')
    parser.add_argument('--overlap', type=float, default=0.5)
    cli_args = parser.parse_args()

    # Load YAML config
    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    if cli_args.test_ckpt is not None:
        config_dict['test_ckpt'] = cli_args.test_ckpt
    if cli_args.data_root is not None:
        config_dict['data_root'] = cli_args.data_root
    if cli_args.split_name is not None:
        config_dict['split_name'] = cli_args.split_name
    if cli_args.output_dir is not None:
        config_dict['output_dir'] = cli_args.output_dir
    if cli_args.overlap is not None:
        config_dict['overlap'] = cli_args.overlap

    return SimpleNamespace(**config_dict)


# ---------- Geometry Utilities ----------



def get_box_corners_with_yaw(center, size, yaw):
    cx, cy, cz = center
    dx, dy, dz = size
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    local_corners = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ])

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1],
    ])

    rotated = np.dot(local_corners, R.T)
    return rotated + np.array([cx, cy, cz])


def create_box_lineset(corners, color=[1, 0, 0]):
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(edges)
    lineset.colors = o3d.utility.Vector3dVector([color for _ in edges])
    return lineset


def merge_linesets_and_save(linesets, save_path):
    all_points, all_lines, all_colors = [], [], []
    offset = 0

    for ls in linesets:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        cols = np.asarray(ls.colors)

        all_points.append(pts)
        all_lines.append(lines + offset)
        all_colors.append(cols)

        offset += pts.shape[0]

    merged = o3d.geometry.LineSet()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    o3d.io.write_line_set(save_path, merged)
    print(f"Saved predicted boxes to {save_path}")


# ---------- Inference Pipeline ----------

def process_pcd_to_model_input(ply_path, use_color=True, num_points=1000000):
    MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
    plydata = PlyData.read(ply_path)
    vertices = np.stack([plydata['vertex'][k] for k in ('x', 'y', 'z', 'red', 'green', 'blue')], axis=-1)

    if use_color:
        points = vertices[:, :6]
        points[:, 3:] = (points[:, 3:] - MEAN_COLOR_RGB) / 256.0
    else:
        points = vertices[:, :3]

    idxs = np.random.choice(points.shape[0], num_points, replace=(points.shape[0] < num_points))
    points = points[idxs]

    pc_min = points[:, :3].min(axis=0)
    pc_max = points[:, :3].max(axis=0)

    return {
        "point_clouds": torch.from_numpy(points).float().unsqueeze(0).cuda(),
        "point_cloud_dims_min": torch.from_numpy(pc_min).float().unsqueeze(0).cuda(),
        "point_cloud_dims_max": torch.from_numpy(pc_max).float().unsqueeze(0).cuda()
    }


def load_model(args, dataset_config):
    model = build_model(args, dataset_config).cuda()
    checkpoint = torch.load(args.test_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def run_inference(model, input_data):
    with torch.no_grad():
        return model(input_data)


def process_predictions(outputs, threshold=0.1, num_angle_bin=1):
    obj_mask = outputs['outputs']['objectness_prob'] > threshold

    cls_logits = outputs['outputs']['sem_cls_logits'][obj_mask]
    classes = torch.argmax(cls_logits, dim=-1)

    center = outputs['outputs']['center_unnormalized'][obj_mask]
    size = outputs['outputs']['size_unnormalized'][obj_mask]
    logits = outputs['outputs']['angle_logits'][obj_mask]
    #yaw = outputs['outputs']['angle_residual'][obj_mask].squeeze(-1)+0.25* math.pi
    yaw = outputs['outputs']['angle_continuous'][obj_mask]
    
    corners = outputs['outputs']['box_corners'][obj_mask]
    objectness_scores = outputs['outputs']['objectness_prob'][obj_mask]

    return center, size, yaw, classes, objectness_scores#,coners


def run_tta_inference_with_nms(model, full_pcd_path, voxel_size=20.0, overlap=0.5,
                               use_color=True, num_points=100000, iou_threshold=0.1):
    pcd = o3d.io.read_point_cloud(full_pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if use_color else None

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    step = voxel_size * (1 - overlap)

    # X slice starting points
    x_starts = list(np.arange(min_bound[0], max_bound[0] - voxel_size + 1e-3, step))
    if not x_starts:
        x_starts = [min_bound[0]]
    elif x_starts[-1] + voxel_size < max_bound[0]:
        x_starts.append(max_bound[0] - voxel_size)

    # Y slice starting points
    y_starts = list(np.arange(min_bound[1], max_bound[1] - voxel_size + 1e-3, step))
    if not y_starts:
        y_starts = [min_bound[1]]
    elif y_starts[-1] + voxel_size < max_bound[1]:
        y_starts.append(max_bound[1] - voxel_size)

    detections = []

    for x in x_starts:
        for y in y_starts:
            x_min, x_max = x, x + voxel_size
            y_min, y_max = y, y + voxel_size

            mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                   (points[:, 1] >= y_min) & (points[:, 1] < y_max)
            if not np.any(mask):
                continue

            sliced_points = points[mask]
            if colors is not None:
                sliced_colors = colors[mask] * 255.0  # Open3D colors are in [0, 1]
                normalized_colors = (sliced_colors - MEAN_COLOR_RGB) / 256.0
                input_pts = np.hstack([sliced_points, normalized_colors])
            else:
                input_pts = sliced_points

            if len(input_pts) < 10:
                continue  # Ignore patches with too few points

            idxs = np.random.choice(len(input_pts), num_points, replace=(len(input_pts) < num_points))
            sampled_points = input_pts[idxs]

            pc_min = sampled_points[:, :3].min(axis=0)
            pc_max = sampled_points[:, :3].max(axis=0)

            input_data = {
                "point_clouds": torch.from_numpy(sampled_points).float().unsqueeze(0).cuda(),
                "point_cloud_dims_min": torch.from_numpy(pc_min).float().unsqueeze(0).cuda(),
                "point_cloud_dims_max": torch.from_numpy(pc_max).float().unsqueeze(0).cuda()
            }
            outputs = run_inference(model, input_data)
            centers, sizes, yaws, classes, objectness_scores = process_predictions(outputs)

            detections.append((centers.cpu(), sizes.cpu(), yaws.cpu(), classes.cpu(),objectness_scores.cpu()))

    return apply_nms_to_detections(detections, iou_threshold=iou_threshold)


def save_predictions_to_ply(center_list, size_list, yaw_list, out_path="output_boxes.ply"):
    linesets = []
    for c, s, y in zip(center_list, size_list, yaw_list):
        corners = get_box_corners_with_yaw(c.cpu().numpy(), s.cpu().numpy(), y.cpu().numpy())
        lineset = create_box_lineset(corners, color=[1, 0, 0])
        linesets.append(lineset)
    merge_linesets_and_save(linesets, out_path)

def save_submission_txt(centers, sizes, yaws, classes,scores, out_path="track1.txt",
                        scene_id=1, frame_id=0):
    """
    Save predictions in the official submission format.

    Format:
    〈scene_id〉 〈class_id〉 〈object_id〉 〈frame_id〉 〈x〉 〈y〉 〈z〉 〈width〉 〈length〉 〈height〉 〈yaw〉 <score>

    Parameters:
        centers (Tensor): (N, 3) tensor of box centers
        sizes (Tensor): (N, 3) tensor of box sizes (dx, dy, dz)
        yaws (Tensor): (N,) tensor of yaw angles
        classes (Tensor): (N,) tensor of class indices
        out_path (str): Output file path
        scene_id (int): Scene ID
        frame_id (int): Frame ID
    """
    with open(out_path, 'w') as f:
        for idx, (c, s, y, cls_id,score) in enumerate(zip(centers, sizes, yaws, classes, scores)):
            c = c.cpu().numpy()
            s = s.cpu().numpy()
            y = y.item()
            cls_id = cls_id.item()
            object_id = -1

            line = f"{scene_id} {cls_id} {object_id} {frame_id} " \
                   f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} " \
                   f"{s[0]:.4f} {s[1]:.4f} {s[2]:.4f} {y:.4f} {score:.4f}\n"
            f.write(line)

    print(f"Submission file saved to {out_path}")

# ---------- Entry Point ----------

def main():
    #parser = make_args_parser()
    args = load_config()

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    datasets, dataset_config = build_dataset(args)
    model = load_model(args, dataset_config)

    
    split_name = args.split_name
    if split_name not in ['val', 'test']:
        raise ValueError(f"Invalid split name: {split_name}. Must be 'val' or 'test'.")
    if split_name == 'test':
        scene_names = ['Warehouse_017', 'Warehouse_018','Warehouse_019','Warehouse_020']
    else:
        scene_names = ['Hospital_000','Lab_000','Warehouse_015','Warehouse_016']
    
    for scene_id, scene_name in enumerate(scene_names):
        for frame_idx in tqdm.tqdm(range(0, 9000), desc="Processing frames"):
            output_path = f'{args.output_dir}/{split_name}_det_out/{scene_name}_{frame_idx:05d}.txt'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if os.path.exists(output_path):
                continue
            with open(output_path, 'w') as f:
                f.write(f"# {scene_name} frame {frame_idx}\n")
            centers, sizes, yaws, classes, scores = run_tta_inference_with_nms(
                model,
                f'{args.data_root}/{split_name}/pcd/{scene_name}_{frame_idx:05d}.ply',
                voxel_size=20.0,
                overlap=args.overlap,
                use_color=args.use_color,
                num_points=args.num_points,
                iou_threshold=0.1
            )

            # Save
            save_submission_txt(centers, sizes, yaws, classes,scores, out_path=output_path,
                                scene_id=scene_id, frame_id=frame_idx)



if __name__ == "__main__":
    main()
