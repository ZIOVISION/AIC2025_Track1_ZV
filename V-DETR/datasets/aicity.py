import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from utils.box_util import (get_3d_box_batch_np, get_3d_box_batch_tensor)
import math
import random

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
IGNORE_LABEL = -100

class AiCityDatasetConfig(object):
    def __init__(self,is_merge_cls=True):
        if is_merge_cls:
            self.num_semcls = 4
            self.class_names = ["Person_FourierGR1T2_AgilityDigit" , "Forklift", "NovaCarter", "Transporter"]
        else:
            self.num_semcls = 6
            self.class_names = ["Person", "Forklift", "NovaCarter", "Transporter", "FourierGR1T2", "AgilityDigit"]
        self.type2class = {name: i for i, name in enumerate(self.class_names)}
        self.class2type = {i: name for i, name in enumerate(self.class_names)}
        self.mean_size_arr = np.ones((self.num_semcls, 3))  # dummy, can be computed
        self.max_num_obj = 128
        self.num_angle_bin = 1



    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_unnorm)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_unnorm)
        return boxes


    def angle2class(self, angle):
        """ angle -> angle_class, angle_residual """
        # num_angle_bin이 1이므로 angle_class는 항상 0, residual은 전체 각도
        angle_class = 0
        angle_residual = angle
        return angle_class, angle_residual

    def class2angle(self, pred_cls, residual):
        """ angle_class, angle_residual -> angle """
        return residual

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return residual

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        return residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual, box_size=None):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb


class AiCityDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        use_height=False,
        augment=False,
        use_random_cuboid=False,
        random_cuboid_min_points=30000,
        args=None,
    ):
        assert split_set in ["train", "val"]
        self.dataset_config = dataset_config
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.args = args

        root_dir = args.dataset_root_dir
        self.split_dir = os.path.join(root_dir, split_set)
        self.pcd_dir = os.path.join(self.split_dir, "pcd")
        self.gt_dir = os.path.join(self.split_dir, "gt")

        self.pcd_files = sorted([f for f in os.listdir(self.pcd_dir) if f.endswith(".ply")])
        self.num_points = args.num_points
        self.use_color = args.use_color
        self.color_mean = args.color_mean

    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, idx):
        file_name = self.pcd_files[idx]
        scan_id = file_name.replace(".ply", "")

        ply_path = os.path.join(self.pcd_dir, file_name)
        gt_path = os.path.join(self.gt_dir, f"{scan_id}.txt")

        plydata = PlyData.read(ply_path)
        vertices = np.stack([plydata['vertex'][k] for k in ('x', 'y', 'z', 'red', 'green', 'blue')], axis=-1)

        if self.use_color:
            points = vertices[:, :6]
            points[:, 3:] = (points[:, 3:] - MEAN_COLOR_RGB) / 256.0
        else:
            points = vertices[:, :3]

        if points.shape[0] >= self.num_points:
            idxs = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            idxs = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[idxs]

        point_cloud_dims_min = points[:, :3].min(axis=0)
        point_cloud_dims_max = points[:, :3].max(axis=0)

        

        # Load GT boxes
        boxes = []
        sem_labels = []
        with open(gt_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                label = int(tokens[0])
                center = list(map(float, tokens[2:5]))
                size = list(map(float, tokens[5:8]))
                yaw = float(tokens[10])
                if yaw < -0.25 * math.pi:
                    yaw += math.pi
                elif yaw >= 3 * math.pi / 4:
                    yaw -= math.pi
                angle = list(map(float, tokens[8:11]))
                angle[2] = yaw
                boxes.append((center, size, angle))
                if self.args.merge_cls:
                    if label==4 or label==5:  # Merge FourierGR1T2 and AgilityDigit
                        label = 0
                sem_labels.append(label)

        num_boxes = len(boxes)
        max_boxes = self.dataset_config.max_num_obj

        target_bboxes = np.zeros((max_boxes, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((max_boxes,), dtype=np.float32)
        angle_classes = np.zeros((max_boxes,), dtype=np.int64)
        angle_residuals = np.zeros((max_boxes,), dtype=np.float32)
        size_residuals = np.zeros((max_boxes, 3), dtype=np.float32)
        box_corners = np.zeros((max_boxes, 8, 3), dtype=np.float32)
        box_centers = np.zeros((max_boxes, 3), dtype=np.float32)
        tmp_angles = np.zeros((max_boxes,), dtype=np.float32)
        
        raw_sizes = np.zeros((max_boxes, 3), dtype=np.float32)
        sem_cls_labels = np.zeros((max_boxes,), dtype=np.int64)

        centers = []
        sizes = []
        angles = []
        

        for i, (center, size, angle) in enumerate(boxes[:max_boxes]):
            target_bboxes[i, 0:3] = center
            target_bboxes[i, 3:6] = size
            target_bboxes_mask[i] = 1
            sem_cls_labels[i] = sem_labels[i]
            centers.append(center)
            sizes.append(size)
            angles.append(angle[2])  # use yaw only
            angle_classes[i], angle_residuals[i] = self.compute_angle_class_and_residual(
                torch.tensor(angle[2]), num_angle_bin=self.dataset_config.num_angle_bin
            )

            raw_sizes[i] = size
            size_residuals[i] = np.array(size) - self.dataset_config.mean_size_arr[sem_labels[i]]

        centers = np.array(centers)
        sizes = np.array(sizes)
        angles = np.array(angles)
        tmp_box_corners = self.dataset_config.box_parametrization_to_corners_np(
            centers, sizes, angles
        )
        box_corners[:len(tmp_box_corners)] = tmp_box_corners

        
        box_centers[:len(centers)] = np.array(centers).astype(np.float32)
        range_diff = point_cloud_dims_max - point_cloud_dims_min
        range_diff[range_diff == 0] = 1e-6

        box_centers_normalized = (box_centers - point_cloud_dims_min) / range_diff
        box_sizes_normalized = raw_sizes / range_diff

        tmp_angles[:len(angles)] = angles
        ret_dict = {
            "file_names": file_name,
            "scan_idx": np.array(idx).astype(np.int64),
            "point_clouds": torch.from_numpy(points).float(),
            "gt_box_corners": box_corners.astype(np.float32),
            "gt_box_centers": box_centers.astype(np.float32),
            "gt_box_sem_cls_label": sem_cls_labels.astype(np.int64),
            "gt_box_present": target_bboxes_mask.astype(np.float32),
            "gt_box_sizes": raw_sizes.astype(np.float32),
            "gt_box_sizes_residual_label": size_residuals.astype(np.float32),
            "gt_angle_class_label": angle_classes.astype(np.int64),
            "gt_angle_residual_label": angle_residuals.astype(np.float32),
            "gt_box_angles": tmp_angles.astype(np.float32),
            "point_cloud_dims_min": point_cloud_dims_min.astype(np.float32),
            "point_cloud_dims_max": point_cloud_dims_max.astype(np.float32),
            "gt_box_centers_normalized": box_centers_normalized.astype(np.float32),
            "gt_box_sizes_normalized": box_sizes_normalized.astype(np.float32),
        }

        return ret_dict


    def collate_fn(self, batch):
        batch_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                batch_dict[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch],dim=0)
            else:
                batch_dict[key] = [sample[key] for sample in batch]

        return batch_dict

    def compute_angle_class_and_residual(self, yaw, num_angle_bin=0):
        device = yaw.device
        angle_class = torch.zeros_like(yaw, dtype=torch.int64, device=device)

        angle_residual = yaw

        return angle_class, angle_residual