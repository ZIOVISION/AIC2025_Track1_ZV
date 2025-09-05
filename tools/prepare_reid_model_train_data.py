import open3d as o3d
import numpy as np
import os
import argparse
from tqdm import tqdm

def crop_and_save_objects_from_pcd(pcd_path, gt_path, output_dir, scene_idx):
    """
    Extract and save point clouds for each object from PLY and GT files.

    Args:
        pcd_path (str): Path to the original PLY file.
        gt_path (str): Path to the Ground Truth text file.
        output_dir (str): Directory to save cropped object files.
        scene_idx (int): Scene index for organizing outputs.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read point cloud file
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"Error reading point cloud file {pcd_path}: {e}")
        return

    # Read Ground Truth file
    try:
        with open(gt_path, 'r') as f:
            gt_data = f.readlines()
    except Exception as e:
        print(f"Error reading ground truth file {gt_path}: {e}")
        return

    # Base filename (without extension)
    base_filename = os.path.splitext(os.path.basename(pcd_path))[0]

    for i, line in enumerate(gt_data):
        parts = line.strip().split()
        if len(parts) != 11:
            print(f"Skipping invalid line in {gt_path}: {line}")
            continue
            
        try:
            obj_class = int(parts[0])
            # Skip certain classes (Person, Forklift, NovaCarter, Transporter)
            if obj_class == 1 or obj_class == 2 or obj_class == 3 or obj_class == 0:
                continue
            
            tmp_scene_idx = scene_idx

            obj_id = int(parts[1])
            # Special handling for FourierGR1T2 and AgilityDigit classes
            if obj_class == 4:  # FourierGR1T2
                obj_id = 9998
            if obj_class == 5:  # AgilityDigit
                obj_id = 9999
                
            center = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            extent = np.array([float(parts[5]), float(parts[6]), float(parts[7])])  # w, l, h
            roll, pitch, yaw = float(parts[8]), float(parts[9]), float(parts[10])
        except ValueError as e:
            print(f"Error parsing line in {gt_path}: {line} - {e}")
            continue

        # Open3D uses RPY (Roll, Pitch, Yaw) to define rotation
        # Create rotation matrix with Euler angles in ZYX order
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))
        
        # Create Oriented Bounding Box
        obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)

        # Find point cloud indices within bounding box
        indices = obb.get_point_indices_within_bounding_box(pcd.points)
        
        # Crop point cloud using indices
        cropped_pcd = pcd.select_by_index(indices)

        if not cropped_pcd.has_points():
            print(f"Warning: No points found for object {obj_id} (class {obj_class}) in {base_filename}.ply")
            continue

        # Save cropped point cloud as a new PLY file
        output_folder = f"{tmp_scene_idx}_{obj_id}"
        os.makedirs(os.path.join(output_dir, output_folder), exist_ok=True)
        existing_files = len(os.listdir(os.path.join(output_dir, output_folder)))
        output_filename = f"{existing_files:05d}.ply"
        output_path = os.path.join(output_dir, output_folder, output_filename)
        o3d.io.write_point_cloud(output_path, cropped_pcd)
        
        print(f"Saved cropped object to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ReID model training data by cropping objects from point clouds")
    parser.add_argument('--pcd_data_root', type=str, 
                       default='../dataset/pcd_dataset',
                       help="Root directory of the PCD dataset")
    parser.add_argument('--out_dir', type=str, 
                       default='../dataset/obj_crop_pcd_dataset',
                       help="Output directory for cropped object point clouds")
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help="Dataset splits to process (space-separated). e.g., --splits train val")
    parser.add_argument('--specific_scene', type=str, default=None,
                       help="Process only a specific scene (e.g., 'Warehouse_014'). If None, process all scenes")
    parser.add_argument('--frame_skip', type=int, default=1,
                       help="Skip every N frames (1 means process all frames)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Starting ReID model training data preparation...")
    print(f"PCD data root: {args.pcd_data_root}")
    print(f"Output directory: {args.out_dir}")
    print(f"Processing splits: {args.splits}")
    print(f"Specific scene: {args.specific_scene if args.specific_scene else 'All scenes'}")
    print(f"Frame skip: {args.frame_skip}")
    
    for split_set in args.splits:
        print(f"\nProcessing split: {split_set}")
        
        split_data_path = os.path.join(args.pcd_data_root, split_set)
        if not os.path.exists(split_data_path):
            print(f"Warning: Split directory {split_data_path} does not exist, skipping...")
            continue
            
        pcd_dir = os.path.join(split_data_path, 'pcd')
        gt_dir = os.path.join(split_data_path, 'gt')
        
        if not os.path.exists(pcd_dir) or not os.path.exists(gt_dir):
            print(f"Warning: PCD or GT directory missing for split {split_set}, skipping...")
            continue
        
        # Get all PCD files
        pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.ply')]
        
        # Filter by specific scene if specified
        if args.specific_scene:
            pcd_files = [f for f in pcd_files if args.specific_scene in f]
        
        # Sort files for consistent processing order
        pcd_files.sort()
        
        # Apply frame skipping
        if args.frame_skip > 1:
            pcd_files = pcd_files[::args.frame_skip]
        
        print(f"Found {len(pcd_files)} files to process in {split_set} split")
        
        # Group files by scene for better progress tracking
        scene_files = {}
        for pcd_file in pcd_files:
            # Extract scene name (assuming format: SceneName_XXXXX.ply)
            scene_name = '_'.join(pcd_file.split('_')[:-1])
            if scene_name not in scene_files:
                scene_files[scene_name] = []
            scene_files[scene_name].append(pcd_file)
        
        # Process each scene
        for scene_idx, (scene_name, files) in enumerate(scene_files.items()):
            print(f"\nProcessing scene: {scene_name} ({len(files)} files)")
            
            # Create output directory for this split
            output_directory = os.path.join(args.out_dir, split_set)
            
            # Process files with progress bar
            for pcd_file in tqdm(files, desc=f"Scene {scene_name}"):
                file_name = os.path.splitext(pcd_file)[0]
                
                # Input file paths
                pcd_file_path = os.path.join(pcd_dir, pcd_file)
                gt_file_path = os.path.join(gt_dir, f'{file_name}.txt')

                # Process if both files exist
                if os.path.exists(pcd_file_path) and os.path.exists(gt_file_path):
                    crop_and_save_objects_from_pcd(pcd_file_path, gt_file_path, output_directory, scene_idx)
                else:
                    if not os.path.exists(pcd_file_path):
                        print(f"Warning: PCD file not found: {pcd_file_path}")
                    if not os.path.exists(gt_file_path):
                        print(f"Warning: GT file not found: {gt_file_path}")

    print("\nReID model training data preparation completed!")


if __name__ == '__main__':
    main()