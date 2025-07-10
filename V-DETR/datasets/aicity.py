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
    def __init__(self):
        self.num_semcls = 6  # 6 classes
        self.class_names = ["Person" , "Forklift", "NovaCarter", "Transporter", "FourierGR1T2", "AgilityDigit"]
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
        # num_angle_bin이 1이므로 residual이 바로 angle이 됨
        return residual

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        # residual 텐서가 바로 각도 값이 됨
        return residual

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        # residual numpy 배열이 바로 각도 값이 됨
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

        root_dir = args.dataset_root_dir  # 예: "/perception/dataset/PhysicalAI-SmartSpaces/pcd_toy_dataset"
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

        # Augmentation
        # if self.augment:
        #     # Random rotation around Z-axis (yaw)
        #     rot_angle = np.random.uniform(-np.pi, np.pi)
        #     rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
        #                         [np.sin(rot_angle), np.cos(rot_angle), 0],
        #                         [0, 0, 1]])

        #     points[:, 0:3] = np.dot(points[:, 0:3], np.transpose(rot_mat))
        #     for i in range(len(boxes)):
        #         center, size, angle = boxes[i]
        #         center = np.dot(center, np.transpose(rot_mat))
        #         angle[2] += rot_angle
        #         while(angle[2] < -0.25 * math.pi):
        #             angle[2] += math.pi
        #         while(angle[2] >= 3 * math.pi / 4):
        #             angle[2] -= math.pi
        #         boxes[i] = (center, size, angle)
            
        #     # 2. 객체 포인트 드롭 (각 박스마다 0~10% 랜덤 드롭)
        #     if len(boxes) > 0:
        #         # (M, N) 형태의 마스크: M은 박스 개수, N은 포인트 개수
        #         boxes = np.array(boxes)
        #         gt_centers = np.array([box[0] for box in boxes])
        #         gt_sizes = np.array([box[1] for box in boxes])
        #         gt_angles = np.array([box[2] for box in boxes])
        #         box_masks = self.points_in_box_mask(points[:, :3], gt_centers, gt_sizes, gt_angles)
                
        #         # 최종적으로 제거할 모든 포인트의 인덱스를 저장할 집합(set)
        #         all_indices_to_drop = set()

        #         # 각 박스를 순회하는 for 루프
        #         for i in range(len(boxes)):
        #             # 현재 박스(i) 내부에 있는 포인트들의 인덱스를 찾음
        #             inside_this_box_indices = np.where(box_masks[i])[0]
                    
        #             # 현재 박스에서 드롭할 포인트가 있다면
        #             if len(inside_this_box_indices) > 0:
        #                 # 드롭할 비율을 0% ~ 10% 사이에서 *새로* 랜덤하게 선택
        #                 drop_ratio = np.random.uniform(0.0, 0.1)
                        
        #                 # 위 비율에 따라 실제 드롭할 포인트 개수 계산
        #                 num_to_drop = int(len(inside_this_box_indices) * drop_ratio)
                        
        #                 if num_to_drop > 0:
        #                     # 드롭할 포인트의 인덱스를 랜덤하게 선택
        #                     indices_to_drop_from_this_box = np.random.choice(
        #                         inside_this_box_indices, num_to_drop, replace=False
        #                     )
        #                     # 최종 제거 리스트에 추가
        #                     all_indices_to_drop.update(indices_to_drop_from_this_box)

        #         # 제거할 포인트가 있다면, 전체 포인트에서 한 번에 제거
        #         if all_indices_to_drop:
        #             # 집합을 리스트로 변환
        #             final_indices_to_drop = list(all_indices_to_drop)
                    
        #             # 남길 포인트들의 마스크 생성 (True는 남김, False는 제거)
        #             keep_mask = np.ones(points.shape[0], dtype=bool)
        #             keep_mask[final_indices_to_drop] = False
        #             points = points[keep_mask]


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
        range_diff[range_diff == 0] = 1e-6  # 나누기 0 방지

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

    def points_in_box_mask(self, points, centers, sizes, angles):
        """
        주어진 포인트들이 3D 바운딩 박스 내부에 있는지 확인합니다.
        
        Args:
            points: (N, 3) 형태의 포인트 배열.
            centers: (M, 3) 형태의 박스 중심점 배열.
            sizes: (M, 3) 형태의 박스 크기(l, w, h) 배열.
            angles: (M,) 형태의 박스 yaw 각도 배열.
            
        Returns:
            (M, N) 형태의 boolean 배열, mask[i, j]는 j번째 포인트가 i번째 박스 안에 있으면 True.
        """
        num_boxes = centers.shape[0]
        num_points = points.shape[0]
        masks = np.zeros((num_boxes, num_points), dtype=bool)

        for i in range(num_boxes):
            center = centers[i]
            size = sizes[i]
            angle = angles[i][2]
            
            # 포인트를 박스의 로컬 좌표계로 변환 (원점을 박스 중심으로 이동)
            translated_points = points - center
            
            # 박스의 yaw 각도만큼 포인트를 반대로 회전
            rot_mat = np.array([[np.cos(-angle), -np.sin(-angle), 0],
                                [np.sin(-angle), np.cos(-angle), 0],
                                [0, 0, 1]])
            
            rotated_points = np.dot(translated_points, rot_mat.T)
            
            # 로컬 좌표계에서 축에 정렬된 박스 내부에 있는지 확인
            half_size = size / 2.0
            mask_i = (np.abs(rotated_points[:, 0]) < half_size[0]) & \
                     (np.abs(rotated_points[:, 1]) < half_size[1]) & \
                     (np.abs(rotated_points[:, 2]) < half_size[2])
            masks[i] = mask_i
            
        return masks

    def collate_fn(self, batch):
        batch_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                batch_dict[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch],dim=0)
            else:
                batch_dict[key] = [sample[key] for sample in batch]

        return batch_dict

    def compute_angle_class_and_residual(self, yaw, num_angle_bin=0):
        """
        """
        device = yaw.device
        # 각도가 속한 bin index
        angle_class = torch.zeros_like(yaw, dtype=torch.int64, device=device)

        angle_residual = yaw

        return angle_class, angle_residual