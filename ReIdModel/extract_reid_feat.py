import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import glob
import argparse

import configs.baseline as cfg
from model import VoxelFeatureExtractor



def crop_point_cloud_from_box(pcd, box_params):
    cx, cy, cz, w, l, h, yaw = box_params
    center = [cx, cy, cz]
    
    extent = [w, l, h] 

    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    
    cropped_pcd = pcd.crop(obb)
    return cropped_pcd

def pcd_to_tensor(pcd, num_points=5000):
    points_xyz = np.asarray(pcd.points, dtype=np.float32)
    
    xyz_min = points_xyz.min(axis=0)
    points_xyz -= xyz_min
    
    points_rgb = np.asarray(pcd.colors, dtype=np.float32)
    combined_features = np.concatenate((points_xyz, points_rgb), axis=1)

    if len(combined_features) == 0:
        return torch.zeros(num_points, 6)
        
    if len(combined_features) < num_points:
        choice = np.random.choice(len(combined_features), num_points, replace=True)
    else:
        choice = np.random.choice(len(combined_features), num_points, replace=False)
    
    final_features = combined_features[choice, :]
    
    return torch.from_numpy(final_features)

def main():
    parser = argparse.ArgumentParser(description="Crop objects, extract ReID features, and save them.")
    parser.add_argument('--model_path', type=str, default="map840.pth", help="Path to the trained model weights.")
    
    
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE)

    model = VoxelFeatureExtractor().to(device)
    model.eval()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    
    
    for split_set in ['test']:
        pcd_dir = f"/perception/dataset/PhysicalAI-SmartSpaces/pcd_dataset/{split_set}/pcd"
        det_dir = f"/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025_outputs/detection/{split_set}_det_out"
        det_files = sorted(glob.glob(os.path.join(det_dir, "*.txt")))

        output_dir = f"/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025_outputs/reid_feat/test_reid_feat_out"
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for det_path in tqdm(det_files, desc="Processing Frames"):
                output_path = os.path.join(output_dir, f"{os.path.basename(det_path).replace('.txt', '.pt')}")
                if os.path.exists(output_path):
                    continue
                with open(output_path, 'w') as f:
                    f.write("# start of feature file\n")

                base_name = os.path.basename(det_path).replace('.txt', '.ply')
                pcd_path = os.path.join(pcd_dir, base_name)

                if not os.path.exists(pcd_path):
                    print(f"Warning: Corresponding PCD file not found for {det_path}. Skipping.")
                    continue

                scene_pcd = o3d.io.read_point_cloud(pcd_path)
                detections = np.loadtxt(det_path)
                if detections.ndim == 1:
                    detections = detections.reshape(1, -1)
                
                frame_features = {}
                
                tensor_list = []
                row_idx_list = []
                for row_idx, det in enumerate(detections):
                    score = det[11]

                    box_params = det[4:11]
                    cropped_pcd = crop_point_cloud_from_box(scene_pcd, box_params)
                    
                    if len(cropped_pcd.points) < 5:
                        continue

                    tensor = pcd_to_tensor(cropped_pcd, num_points=cfg.NUM_POINTS)
                    
                    tensor_list.append(tensor)
                    row_idx_list.append(row_idx)
                    if len(tensor_list) >= 128:
                        batch_tensors = torch.stack(tensor_list).to(device)
                        embeddings = model(batch_tensors)
                        
                        for idx, embedding in zip(row_idx_list, embeddings):
                            feature_key = f"row_{idx}_obj"
                            frame_features[feature_key] = embedding.cpu().squeeze()
                        
                        tensor_list.clear()
                        row_idx_list.clear()
                if len(tensor_list) > 0:
                    batch_tensors = torch.stack(tensor_list).to(device)
                    embeddings = model(batch_tensors)
                    for idx, embedding in zip(row_idx_list, embeddings):
                        feature_key = f"row_{idx}_obj"
                        frame_features[feature_key] = embedding.cpu().squeeze()
                
                torch.save(frame_features, output_path)
                
        print("\nFeature extraction complete!")
        print(f"Saved features can be found in: {output_dir}")

if __name__ == "__main__":
    main()