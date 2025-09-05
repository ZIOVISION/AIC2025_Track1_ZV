#!/bin/bash
CUR_DIR=$(pwd)

cd ReIdModel

python extract_reid_feat.py \
    --split_name "test" \
    --det_root "${CUR_DIR}/outputs/detection" \
    --reid_feat_root "${CUR_DIR}/outputs/reid_feat" \
    --data_root "${CUR_DIR}/dataset/pcd_dataset"

cd "${CUR_DIR}"