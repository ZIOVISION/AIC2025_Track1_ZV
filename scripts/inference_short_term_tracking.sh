#!/bin/bash
CUR_DIR=$(pwd)

cd Tracker

python short_term_tracking.py \
    --split_name "test" \
    --detection_folder "${CUR_DIR}/outputs/detection" \
    --use_reid_feat \
    --reid_feat_folder "${CUR_DIR}/outputs/reid_feat" \
    --output_folder "${CUR_DIR}/outputs/tracking"

cd "${CUR_DIR}"