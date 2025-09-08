#!/bin/bash
CUR_DIR=$(pwd)

cd ReIdModel

python extract_reid_feat.py \
    --model_path "${CUR_DIR}/weights/best_reid_model.pth" \
    --split_name "test" \
    --det_root "${CUR_DIR}/outputs/detection" \
    --output_dir "${CUR_DIR}/outputs/reid_feat" \
    --data_root "${CUR_DIR}/dataset/pcd_dataset"

cd "${CUR_DIR}"