import os
import glob
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict



def trajectory_matching(chunk_idx, track_res_folder, reid_feat_folder, output_folder):
    """
    비디오 내에서 궤적을 재연결하고, 최종적으로 합쳐진 궤적의 길이를 기준으로
    다시 필터링하여 결과를 저장하는 최종 함수.
    """
    # --- 1. Configuration ---
    print("--- 1. Configuration initialization ---")
    print(f"Track results folder: {track_res_folder}")
    print(f"ReID features folder: {reid_feat_folder}")
    print(f"Output folder: {output_folder}")
    
    frames_per_chunk = 9000

    MIN_TRAJ_LENGTH = 60
    TIME_THRESHOLD = 600
    DIST_THRESHOLD = 3.0
    FEAT_THRESHOLD = 0.6
    
    # --- 2. Trajectory information aggregation ---
    print("\n--- 2. Starting trajectory information aggregation ---")
    
    all_track_files = sorted(glob.glob(os.path.join(track_res_folder, '*.txt')))


    trajectories = {}
    
    print(f"  Loading chunk {chunk_idx} data...")
    chunk_files = all_track_files[frames_per_chunk*chunk_idx : frames_per_chunk*(chunk_idx+1)]
    
    for file_path in chunk_files:
        file_name = os.path.basename(file_path).split('.')[0]
        feat_file_path = os.path.join(reid_feat_folder, f"{file_name}.pt")
        all_obj_features = torch.load(feat_file_path)

        with open(file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                cls_id, local_track_id, global_frame_id = int(parts[1]), int(parts[2]), int(parts[3])
                box = np.array(parts[4:11], dtype=float)
                row_idx = int(parts[12])
                
                traj_key = local_track_id
                if traj_key not in trajectories:
                    trajectories[traj_key] = {'features':[], 'frames':[], 'boxes':[], 'classes':[]}
                
                feature = all_obj_features.get(f'row_{row_idx}_obj')
                if feature is not None:
                    trajectories[traj_key]['features'].append(feature.numpy())
                trajectories[traj_key]['frames'].append(global_frame_id)
                trajectories[traj_key]['boxes'].append(box)
                trajectories[traj_key]['classes'].append(cls_id)

    print("\n  Calculating representative values for each trajectory...")
    for data in trajectories.values():
        if data['features']:
            mean_feature = np.mean(np.array(data['features']), axis=0)
            data['mean_feature'] = mean_feature / np.linalg.norm(mean_feature)
        else:
            data['mean_feature'] = None

        data['dominant_class'] = Counter(data['classes']).most_common(1)[0][0]
        data['first_box'] = data['boxes'][0]
        
        if len(data['boxes']) > 15:
            data['last_box'] = data['boxes'][-15]
        else:
            data['last_box'] = data['boxes'][-1]
        data['first_frame'] = data['frames'][0]
        data['last_frame'] = data['frames'][-1]

    #print(trajectories)

    while True:
        trajectories_keys = list(trajectories.keys())
        trajectories_keys = [int(key) for key in trajectories_keys]
        max_key = max(trajectories_keys) + 1 if trajectories_keys else 0
        cost_matrix = np.zeros((max_key, max_key))+500

        matching_results = {}
        for traj_key1, data1 in trajectories.items():
            if len(data1['frames']) < MIN_TRAJ_LENGTH:
                continue
            for traj_key2, data2 in trajectories.items():
                if traj_key1 >= traj_key2:
                    continue
                if len(data2['frames']) < MIN_TRAJ_LENGTH:
                    continue

                
                if data1['dominant_class'] != data2['dominant_class']:
                    continue
                
                if np.linalg.norm(data1['last_box'][:2] - data2['first_box'][:2]) > DIST_THRESHOLD:
                    continue

                if abs(data2['first_frame'] - data1['last_frame'])  > TIME_THRESHOLD:
                    continue

                if data1['mean_feature'] is not None and data2['mean_feature'] is not None:
                    feat_dist = np.linalg.norm(data1['mean_feature'] - data2['mean_feature'])
                    if feat_dist > FEAT_THRESHOLD:
                        continue

                cost_matrix[traj_key1, traj_key2] = abs(data2['first_frame'] - data1['last_frame'])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 500:
                matching_results[r] = {'parent': c, 'dist': cost_matrix[r, c]}


        print('Number of trajectories:', len(trajectories))
        for child,parent_info in reversed(matching_results.items()):
            child_key = child
            parent_key = parent_info['parent']

            print(f"  {child_key} -> {parent_key} (distance: {parent_info['dist']})")
            child_key = int(child_key)
            parent_key = int(parent_key)


            trajectories[child_key]['frames'].extend(trajectories[parent_key]['frames'])
            trajectories[child_key]['boxes'].extend(trajectories[parent_key]['boxes'])
            trajectories[child_key]['classes'].extend(trajectories[parent_key]['classes'])

            # Remove duplicate frames (also remove corresponding boxes and classes)
            unique_frames = set(trajectories[child_key]['frames'])
            unique_indices = [trajectories[child_key]['frames'].index(frame) for frame in unique_frames]
            trajectories[child_key]['frames'] = [trajectories[child_key]['frames'][i] for i in unique_indices]
            trajectories[child_key]['boxes'] = [trajectories[child_key]['boxes'][i] for i in unique_indices]
            trajectories[child_key]['classes'] = [trajectories[child_key]['classes'][i] for i in unique_indices]

            trajectories[child_key]['last_box'] = trajectories[child_key]['boxes'][-1]
            trajectories[child_key]['last_frame'] = trajectories[child_key]['frames'][-1]
            trajectories[child_key]['first_box'] = trajectories[child_key]['boxes'][0]
            trajectories[child_key]['first_frame'] = trajectories[child_key]['frames'][0]
            
            


            # Remove the parent trajectory
            del trajectories[parent_key]

        print("\n  Final trajectory count:", len(trajectories))
        if len(matching_results) == 0:
            break
    
    results_lines = []
    for trajectory in trajectories:
        traj_data = trajectories[trajectory]
        unique_frames = set(traj_data['frames'])
        unique_indices = [traj_data['frames'].index(frame) for frame in unique_frames]
        traj_data['frames'] = [traj_data['frames'][i] for i in unique_indices]
        traj_data['boxes'] = [traj_data['boxes'][i] for i in unique_indices]
        traj_data['classes'] = [traj_data['classes'][i] for i in unique_indices]
        frames = traj_data['frames']
        boxes = traj_data['boxes']
        classes = traj_data['classes']


        
        if len(frames) < MIN_TRAJ_LENGTH:
            continue

        for frame, box, cls in zip(frames, boxes, classes):
            cx, cy, cz, w, l, h, yaw = box
            global_id = trajectory  # Use local ID as global ID
            scene_id = chunk_idx+17  # Fixed value for example, adjust as needed in practice
            cls = traj_data['dominant_class']  # Unify with dominant class
            results_lines.append(f"{scene_id} {cls} {global_id} {frame} {cx:.2f} {cy:.2f} {cz:.2f} {w:.2f} {l:.2f} {h:.2f} {yaw:.2f}\n")

    return results_lines 
   
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Long-term trajectory matching and re-identification")
    parser.add_argument("--split_name", type=str, default="test", choices=["test", "val"], 
                       help="Dataset split to process")
    parser.add_argument("--track_res_folder", type=str, 
                       default="./outputs/tracking", 
                       help="Base folder containing short-term tracking results")
    parser.add_argument("--reid_feat_folder", type=str, 
                       default="./outputs/reid_feat", 
                       help="Base folder containing ReID features")
    parser.add_argument("--output_folder", type=str, 
                       default="./outputs/tracking", 
                       help="Base output folder for long-term tracking results")
    args = parser.parse_args()

    # Construct split-specific paths
    track_res_folder = os.path.join(args.track_res_folder, f"{args.split_name}_track_out")
    reid_feat_folder = os.path.join(args.reid_feat_folder, f"{args.split_name}_reid_feat_out")
    output_folder = os.path.join(args.output_folder, f"{args.split_name}_long_term_out")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Create final combined output file
    final_output_path = os.path.join(output_folder, f'long_term_{args.split_name}_all.txt')
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
    
    # Process all chunks (0, 1, 2, 3)
    for chunk_idx in range(4):
        print(f"\nProcessing chunk {chunk_idx}...")
        res = trajectory_matching(chunk_idx, track_res_folder, reid_feat_folder, output_folder)
        
        # Save results for this chunk
        chunk_output_path = os.path.join(output_folder, f'long_term_chunk_{chunk_idx}.txt')
        with open(chunk_output_path, 'w') as f:
            f.writelines(res)
        print(f"Chunk {chunk_idx} results saved to {chunk_output_path}")
        
        # Append to final combined file
        with open(final_output_path, 'a') as f:
            f.writelines(res)
    
    print(f"\nAll chunks processed. Final combined results saved to {final_output_path}")