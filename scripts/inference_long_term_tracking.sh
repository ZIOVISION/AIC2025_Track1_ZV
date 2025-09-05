#!/bin/bash
CUR_DIR=$(pwd)

cd Tracker

python long_term_tracking.py \
    --split_name "test" \
    --track_res_folder "${CUR_DIR}/outputs/tracking" \
    --reid_feat_folder "${CUR_DIR}/outputs/reid_feat" \
    --output_folder "${CUR_DIR}/outputs/tracking"

cd "${CUR_DIR}"

echo "Long-term tracking completed for all chunks."