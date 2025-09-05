import sys
import os
import argparse
import json
from multiprocessing import freeze_support

# TrackEval 라이브러리 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval

OBJECT_CLASSES = ['Person', 'Forklift', 'NovaCarter', 'Transporter', 'FourierGR1T2', 'AgilityDigit']
output_path = 'ground_truth.txt'
#gt file 
scene_names = ['Hospital_000', 'Lab_000','Warehouse_015', 'Warehouse_016']
with open(output_path, 'w') as out_file:
    for scene_idx, scene_name in enumerate(scene_names):
        gt_path = f'/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025/val/{scene_name}/ground_truth.json'
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        frame_list = gt_data.keys()
        for frame in frame_list:
            frame_data = gt_data[frame]
            for obj in frame_data:
                obj_id = obj['object id']
                location = obj['3d location']
                scale = obj['3d bounding box scale']
                yaw = obj['3d bounding box rotation'][2]
                cls_name = obj['object type']
                cls_idx = OBJECT_CLASSES.index(cls_name) if cls_name in OBJECT_CLASSES else -1
                
                # Write to output file
                
                out_file.write(f"{scene_idx} {cls_idx} {obj_id} {frame} {location[0]} {location[1]} {location[2]} {scale[0]} {scale[1]} {scale[2]} {yaw}\n")


def main():
    freeze_support()

    gt_file_path = 'ground_truth.txt'
    prediction_file_path = '/perception/heefe/git_projects/AIC2025_Track1_ZV/Tracker/best_track1.txt'
    
    classes_to_evaluate = ['Person', 'Forklift', 'NovaCarter', 'Transporter', 'FourierGR1T2']#, 'AgilityDigit']
    # -----------------------

    # 평가기 설정
    eval_config = {
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': os.cpu_count(),
    }
    evaluator = trackeval.Evaluator(eval_config)

    # 데이터셋 설정
    dataset_config = {
        'GT_FILE_PATH': gt_file_path,
        'TRACKER_FILE_PATH': prediction_file_path,
        'CLASSES_TO_EVAL': classes_to_evaluate,
    }
    # 새로 만든 데이터셋 클래스를 사용
    dataset_list = [trackeval.datasets.AICityChallengeSingleFile(dataset_config)]
    
    # 평가 지표 설정 (HOTA가 기본)
    metrics_list = [
        trackeval.metrics.HOTA(), 
    ]

    # 평가 실행 및 결과 출력
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    
    # 상세 결과 출력
    print("\n--- Evaluation Results ---")
    for dataset, res_by_class in output_res.items():
        for class_name, metrics in res_by_class['prediction'].items():
            print(f"\n===== Class: {class_name} =====")
            for metric_name, value in metrics.items():
                if metric_name == 'HOTA':
                    # HOTA, CLEAR, Identity 지표만 출력
                    print(f"{metric_name}: {value:.3f}")
                

if __name__ == '__main__':
    # shapely 라이브러리가 설치되어 있는지 확인
    try:
        import shapely
    except ImportError:
        print("\n[ERROR] Shapely library not found.")
        print("Please install it using: pip install shapely\n")
        sys.exit(1)
        
    main()