import os
import glob
import numpy as np

from tracker import DeepSort3D
import torch

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Short Term Tracking with DeepSort3D")
    parser.add_argument('--split_name', type=str, default='test', choices=['test', 'val'], 
                       help="Dataset split to process")
    parser.add_argument('--detection_folder', type=str, default='./outputs/detection', 
                       help="Base folder containing detection results")
    parser.add_argument('--use_reid_feat', action='store_true', default=False, 
                       help="Use reid features for tracking")
    parser.add_argument('--reid_feat_folder', type=str, default='./outputs/reid_feat', 
                       help="Base folder containing reid feature .pt files")
    parser.add_argument('--output_folder', type=str, default='./outputs/tracking', 
                       help="Base folder to save tracking results .txt files")
    args = parser.parse_args()
    
    
    # Construct split-specific paths
    if args.split_name == 'test':
        detection_folder = os.path.join(args.detection_folder, f"{args.split_name}_det_out_refine_cls")
    else:  # val
        detection_folder = os.path.join(args.detection_folder, f"{args.split_name}_det_out_refine_cls")
    
    reid_feat_folder = os.path.join(args.reid_feat_folder, f"{args.split_name}_reid_feat_out")
    output_folder = os.path.join(args.output_folder, f"{args.split_name}_track_out")
    use_reid_feat = args.use_reid_feat
    
    print(f"Processing {args.split_name} split...")
    print(f"Detection folder: {detection_folder}")
    print(f"ReID features folder: {reid_feat_folder}")
    print(f"Output folder: {output_folder}")
    
    MAX_AGE = 15
    MIN_HITS = 1
    IOU_THRESHOLD = 0.4
    CONF_THRESHOLD = 0.35

    # --- Prepare folders ---
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder '{output_folder}'")

    all_detection_files = sorted(glob.glob(os.path.join(detection_folder, '*.txt')))
    if use_reid_feat:
        all_feat_files = sorted(glob.glob(os.path.join(reid_feat_folder, '*.pt')))
    for i in range(0,4):
        # --- Run tracker ---
        tracker = DeepSort3D(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
        
        
        detection_files = all_detection_files[9000*i:9000*(i+1)]  # Split into 9000 files each
        if use_reid_feat:
            feat_files = all_feat_files[9000*i:9000*(i+1)]

        
        for frame_num, file_path in enumerate(detection_files):
            print(f"\n--- Frame {frame_num} ({os.path.basename(file_path)}) ---")
            if use_reid_feat:
                all_obj_features = torch.load(feat_files[frame_num])
            
 
            data = np.loadtxt(file_path)
            if data.ndim == 1: data = data.reshape(1, -1)
            
            scene_id = int(data[0, 0])
            
            # Extract necessary columns: cx, cy, cz, w, l, h, yaw, score, cls_id
            detections = data[:, [4, 5, 6, 7, 8, 9, 10, 11, 1]]
            detections = detections.tolist()  # Convert to list
            for row_idx in range(len(detections)):
                detections[row_idx].append(row_idx) # Add row index

            detections = np.array(detections, dtype=np.float32)

            # Exclude objects with low confidence score
            detections = detections[detections[:, 7] >= CONF_THRESHOLD]


            delete_idx = []
            
            if use_reid_feat:
                for idx, det in enumerate(detections):
                    row_idx = int(det[9])
                    if f'row_{row_idx}_obj' not in all_obj_features:
                        delete_idx.append(idx)
            
            detections = np.delete(detections, delete_idx, axis=0)  # Remove corresponding rows
            if use_reid_feat:
                features = []
                for det in detections:
                    row_idx = int(det[9])
                    feat = all_obj_features[f'row_{row_idx}_obj']
                    features.append(feat.numpy())  # Convert torch tensor to numpy
                features = np.array(features, dtype=np.float32)
                

            if use_reid_feat:
                # Update tracker
                tracked_objects = tracker.update(detections,features)
            else:
                # Update tracker (without ReID features)
                tracked_objects = tracker.update(detections)
            
            # Display results on screen (same as before)
            print("Tracking results:")
            if tracked_objects.size > 0:
                print(" cx      cy     cz      w      l      h     yaw    score  ID  CLS")
                print("-" * 71)
                for obj in tracked_objects:
                    print(f"{obj[0]: 7.2f} {obj[1]: 7.2f} {obj[2]: 7.2f} {obj[3]: 6.2f} {obj[4]: 6.2f} {obj[5]: 6.2f} {obj[6]: 7.2f} {obj[7]: 7.2f} {int(obj[8]):>3d} {int(obj[9]):>4d}")
            else:
                print("No tracked objects.")

            # --- Save result file (added part) ---
            output_filename = os.path.basename(file_path)
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, 'w') as f:
                if tracked_objects.size > 0:
                    # tracked_objects format: [cx, cy, cz, w, l, h, yaw, score, track_id, cls_id]
                    for obj in tracked_objects:
                        cx, cy, cz, w, l, h, yaw = obj[0:7]
                        score = obj[7]
                        track_id = int(obj[8])
                        cls_id = int(obj[9])
                        row_idx = int(obj[10])
                        
                        # Output format: {scene_id} {cls_id} {object_id} {frame_id} {cx} ... {score}
                        # Use the track_id assigned to object_id
                        output_line = (f"{scene_id} {cls_id} {track_id} {frame_num} "
                                    f"{cx:.4f} {cy:.4f} {cz:.4f} {w:.4f} {l:.4f} {h:.4f} {yaw:.4f} {score:.4f} {row_idx}\n")
                        f.write(output_line)
            
            print(f"Tracking results saved to '{output_path}'.")