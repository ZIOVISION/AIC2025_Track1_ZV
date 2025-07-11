import os
import glob
from collections import defaultdict
import random

import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

def create_query_gallery_split(data_path):
    print(f"\nDynamically splitting {data_path} into query and gallery for validation...")
    all_files_by_pid = defaultdict(list)
    obj_folders = sorted(glob.glob(os.path.join(data_path, "*")))
    unique_obj_ids = {name: i for i, name in enumerate(sorted([os.path.basename(p) for p in obj_folders]))}
    for obj_folder in obj_folders:
        folder_name = os.path.basename(obj_folder)
        if folder_name not in unique_obj_ids: continue
        pid = unique_obj_ids[folder_name]
        ply_files = glob.glob(os.path.join(obj_folder, "*.ply"))
        for ply_file in ply_files:
            all_files_by_pid[pid].append(ply_file)
    
    query_data, gallery_data = [], []
    for pid, files in all_files_by_pid.items():
        if not files: continue
        for file_path in files:
            gallery_data.append((file_path, pid))
        query_file = random.choice(files)
        query_data.append((query_file, pid))
        
    print(f"Validation split complete. Query size: {len(query_data)}, Gallery size: {len(gallery_data)}")
    return query_data, gallery_data

class ListBasedReIDDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        file_path, pid = self.data_list[index]
        point_cloud = read_ply_to_tensor(file_path)
        return point_cloud, pid

class RandomRotate(object):
    def __init__(self, axis='z', max_angle=180):
        self.axis = axis
        self.max_angle = np.deg2rad(max_angle)

    def __call__(self, point_cloud):
        points = point_cloud[:, :3]
        colors = point_cloud[:, 3:]
        
        angle = random.uniform(-self.max_angle, self.max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        if self.axis == 'x':
            R = torch.tensor([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=torch.float32)
        elif self.axis == 'y':
            R = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=torch.float32)
        else:
            R = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32)

        rotated_points = points @ R.T
        return torch.cat((rotated_points, colors), dim=1)

class ColorJitter(object):
    def __init__(self, strength=0.1):
        self.strength = strength

    def __call__(self, point_cloud):
        points = point_cloud[:, :3]
        colors = point_cloud[:, 3:]
        noise = torch.empty_like(colors).uniform_(-self.strength, self.strength)
        jittered_colors = torch.clamp(colors + noise, 0.0, 1.0)
        return torch.cat((points, jittered_colors), dim=1)

def read_ply_to_tensor(path, num_points=5000):
    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return torch.zeros(num_points, 6)

    points_xyz = np.asarray(pcd.points, dtype=np.float32)
    if points_xyz.shape[0] == 0:
        return torch.zeros(num_points, 6)

    xyz_min = points_xyz.min(axis=0)
    points_xyz -= xyz_min
    
    points_rgb = np.asarray(pcd.colors, dtype=np.float32)
    if points_rgb.shape[0] == 0:
        points_rgb = np.full_like(points_xyz, 0.5, dtype=np.float32)

    combined_features = np.concatenate((points_xyz, points_rgb), axis=1)

    if len(combined_features) < num_points:
        choice = np.random.choice(len(combined_features), num_points, replace=True)
    else:
        choice = np.random.choice(len(combined_features), num_points, replace=False)
    
    final_features = combined_features[choice, :]
    return torch.from_numpy(final_features)


class PointCloudReIDDataset(Dataset):
    def __init__(self, data_path, use_augmentation=False, num_points=5000):
        super().__init__()
        self.data_path = data_path
        self.num_points = num_points
        
        self.file_paths = []
        self.pids = []
        self.scene_ids = []
        
        self.transform = None
        if use_augmentation:
            self.transform = transforms.Compose([
                RandomRotate(axis='z', max_angle=180),
                RandomRotate(axis='x', max_angle=3),
                RandomRotate(axis='y', max_angle=3),
                ColorJitter(strength=0.2),
            ])

        obj_folders = sorted(glob.glob(os.path.join(data_path, "*")))
        self.unique_obj_ids = {name: i for i, name in enumerate(sorted([os.path.basename(p) for p in obj_folders]))}

        for obj_folder in obj_folders:
            folder_name = os.path.basename(obj_folder)
            scene_idx, _ = folder_name.split('_')
            scene_idx = int(scene_idx)
            
            pid = self.unique_obj_ids[folder_name]
            
            ply_files = glob.glob(os.path.join(obj_folder, "*.ply"))
            for ply_file in ply_files:
                self.file_paths.append(ply_file)
                self.pids.append(pid)
                self.scene_ids.append(scene_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        pid = self.pids[index]
        scene_id = self.scene_ids[index]
        
        point_cloud = read_ply_to_tensor(path, num_points=self.num_points)
        
        if self.transform:
            point_cloud = self.transform(point_cloud)
            
        return point_cloud, pid, scene_id


class SceneBatchSampler(Sampler):
    def __init__(self, dataset, p, k):
        super().__init__(dataset)
        self.dataset = dataset
        self.p = p
        self.k = k
        
        self.scene_to_pids = defaultdict(lambda: defaultdict(list))
        for idx, (pid, scene_id) in enumerate(zip(dataset.pids, dataset.scene_ids)):
            self.scene_to_pids[scene_id][pid].append(idx)
        
        self.scenes = list(self.scene_to_pids.keys())
        
        self.batches_per_scene = {
            scene: len(pids) // self.p for scene, pids in self.scene_to_pids.items()
        }
        self.total_batches = sum(self.batches_per_scene.values())

    def __iter__(self):
        batch_indices = []
        
        for scene_id in self.scenes:
            scene_pids = list(self.scene_to_pids[scene_id].keys())
            
            num_batches_in_scene = len(scene_pids) // self.p
            if num_batches_in_scene == 0:
                continue

            np.random.shuffle(scene_pids)

            for i in range(num_batches_in_scene):
                batch = []
                start = i * self.p
                end = start + self.p
                selected_pids = scene_pids[start:end]
                
                for pid in selected_pids:
                    indices = self.scene_to_pids[scene_id][pid]
                    selected_indices = np.random.choice(indices, self.k, replace=True)
                    batch.extend(selected_indices)
                
                batch_indices.append(batch)

        np.random.shuffle(batch_indices)
        return iter(batch_indices)

    def __len__(self):
        return self.total_batches


def create_dataloader(data_path, p_size=16, k_size=4, num_workers=4, use_augmentation=False):
    dataset = PointCloudReIDDataset(
        data_path=data_path, 
        use_augmentation=use_augmentation
    )
    
    sampler = SceneBatchSampler(dataset=dataset, p=p_size, k=k_size)

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )
    
    num_classes = len(dataset.unique_obj_ids)
    
    return dataloader, num_classes