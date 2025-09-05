#!/bin/bash
CUR_DIR=$(pwd)

cd ReIdModel

python embedding_similarity_classification.py \
    --split_name "test" \
    --det_root "${CUR_DIR}/outputs/detection" \
    --reid_feat_root "${CUR_DIR}/outputs/reid_feat" \
    --model_path "${CUR_DIR}/weights/reid/best_reid_model.pth" \
    --FourierGR1T2_dir "${CUR_DIR}/dataset/obj_crop_pcd_dataset/train/0_9998" \
    --AgilityDigit_dir "${CUR_DIR}/dataset/obj_crop_pcd_dataset/train/0_9999"

cd "${CUR_DIR}"