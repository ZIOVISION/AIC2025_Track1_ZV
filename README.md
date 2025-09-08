# Multi-Camera 3D Object Tracking via V-DETR and Point Cloud Re-Identification

## Overview
This repository contains a solution for Multi-Camera Multi-Target Tracking (MCMT) designed for the [AI City Challenge 2025 Track1](https://www.aicitychallenge.org/2025-track1/). 

Our approach combines:
- **V-DETR**: 3D object detection using DETR with Vertex Relative Position Encoding
- **Point Cloud Re-ID**: 3D point cloud-based re-identification for robust target association
- **Embedding Similarity Classification**: Gallery-based classification refinement using feature similarity matching
- **Multi-Camera Tracking**: Progressive association framework with short-term and long-term trajectory linking


## Dataset Availability

The official dataset can be downloaded from the AI City Challenge website (https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2025). 


## Overall Pipeline

<img src="pipeline.png" width="1050" />
 

## Results

### Competition Performance

Our method achieved **1st place** in the AI City Challenge 2025 Track1:

| Team (Rank) | HOTA ↑ | DetA ↑ | AssA ↑ | LocA ↑ |
|-------------|---------|---------|---------|---------|
| **65 (1st, Ours)** | **69.91** | **71.34** | **69.06** | **78.27** |
| 15 (2nd) | 63.14 | - | - | - |
| 133 (3rd) | 28.75 | - | - | - |

### Ablation Study

Ablation study on the validation set. The performance of our baseline is progressively improved by adding the ESC (+ESC) and the offline Global Tracking framework (+Global Track). We report results on all classes and by excluding the *AgilityDigit* class, which is absent from the validation set.

| Setting | Method | HOTA ↑ | DetA ↑ | AssA ↑ | LocA ↑ |
|---------|---------|---------|---------|---------|---------|
| All Classes | Baseline | 37.86 | 37.71 | 42.12 | 90.78 |
| All Classes | +ESC | 44.68 | 50.55 | 43.49 | **92.06** |
| All Classes | +Global Track | **58.20** | **56.00** | **64.53** | 91.98 |
| w/o AgilityDigit | Baseline | 45.44 | 45.25 | 50.54 | 88.94 |
| w/o AgilityDigit | +ESC | 53.61 | 60.67 | 52.19 | **90.47** |
| w/o AgilityDigit | +Global Track | **69.84** | **67.19** | **77.43** | 90.37 |



## Environment Requirements

Our implementation is built upon:
- **V-DETR**: For 3D object detection
- **MinkowskiEngine**: For sparse 3D convolution operations
- **PyTorch Metric Learning**: For re-identification training
- **TrackEval**: For evaluation metrics

### Installation

**Step 1.** Clone the repository with submodules:
```bash
git clone --recursive https://github.com/ZIOVISION/AIC2025_Track1_ZV.git
cd AIC2025_Track1_ZV
```

**Step 2.** Create a conda environment:
```bash
conda create --name aic2025 python=3.8 -y
conda activate aic2025
```

**Step 3.** Install PyTorch:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
```

**Step 4.** Install required packages:
```bash
cd V-DETR
pip install -r requirements.txt
cd ..
```

**Step 5.** Install MinkowskiEngine:
```bash
cd MinkowskiEngine
conda install -c conda-forge openblas -y
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
cd ..
```

**Step 6.** Install pytorch-metric-learning:
```bash
cd pytorch-metric-learning
pip install -e .
cd ..
```

**Step 7.** Install third party dependencies:
```bash
pip install openmim
mim install mmcv-full==1.6.1
cd V-DETR/third_party/pointnet2/ && python setup.py install --user
cd ../../..
pip install h5py open3d shapely filterpy
cd V-DETR/utils && python cython_compile.py build_ext --inplace
cd ../..
```

### Project Structure
```
AIC2025_Track1_ZV/
│  README.md
│
├─V-DETR/                 # 3D object detection
├─ReIdModel/              # Point cloud re-identification  
├─Tracker/                # Multi-camera tracking
├─TrackEval/              # Evaluation tools
├─MinkowskiEngine/        # Sparse 3D convolution
├─pytorch-metric-learning/# Re-ID training utilities
├─scripts/                # Training and inference scripts
├─tools/                  # Data preparation utilities
├─dataset/                # Dataset directory
└─weights/                # Model weights
```

## Data Preparation

Before training, you need to prepare the dataset and convert it to the required formats.

### 1. Download Dataset
Download the official AI City Challenge 2025 Track1 dataset from Hugging Face:
```bash
python tools/data_download.py --data-root dataset
```

This will download the dataset to `dataset/MTMC_Tracking_2025/` with the following structure:
```
dataset/MTMC_Tracking_2025/
├─train/
│  ├─Warehouse_XXX/
│  │  ├─videos/           # Multi-camera video files (.mp4)
│  │  ├─depth_maps/       # Depth maps (.h5)
│  │  ├─calibration.json  # Camera calibration parameters
│  │  └─ground_truth.json # 3D bounding box annotations
│  └─...
├─val/
└─test/
```

### 2. Generate Point Cloud Data
Convert RGB-D video sequences to 3D point clouds:
```bash
python tools/generate_pcd_data.py \
    --data-root dataset/MTMC_Tracking_2025 \
    --out-dir dataset/pcd_dataset \
    --splits train val test
```

This script:
- Reads multi-camera RGB videos and corresponding depth maps
- Applies camera calibration to generate world-coordinate point clouds
- Combines point clouds from all cameras per frame
- Converts 3D annotations to point cloud coordinate system
- Saves point clouds as `.ply` files and annotations as `.txt` files

### 3. Prepare Training Data
Create training data for different components:

**For V-DETR detector:**
```bash
python tools/prepare_detector_train_data.py \
    --pcd-data-root dataset/pcd_dataset \
    --out-dir dataset/pcd_dataset_slice_20
```

This creates spatially sliced point cloud data (20m×20m patches) suitable for 3D object detection training.

**For Re-ID model:**
```bash
python tools/prepare_reid_model_train_data.py \
    --pcd-data-root dataset/pcd_dataset \
    --out-dir dataset/obj_crop_pcd_dataset
```

This extracts object-centered point cloud patches for re-identification training.

**All-in-one preparation:**
```bash
bash scripts/prepare_datasets.sh
```

## Training Process

Our training pipeline consists of three main components:

### 1. 3D Object Detection (V-DETR)
Train the V-DETR model for 3D object detection:
```bash
bash scripts/train_detector.sh
```

### 2. Point Cloud Re-Identification Model
Train the re-identification model using prepared object-centered point clouds:
```bash
bash scripts/train_reid_model.sh
```

## Inference Pipeline

### Prerequisites
Ensure you have completed the [Data Preparation](#data-preparation) steps above, particularly for the test set:
```bash
python tools/generate_pcd_data.py \
    --data-root dataset/MTMC_Tracking_2025 \
    --out-dir dataset/pcd_dataset \
    --splits test
```

### Complete Pipeline
Run the complete inference pipeline with these scripts (configured for test split by default):

### 1. 3D Object Detection (V-DETR)
Run V-DETR inference on prepared point cloud data:
```bash
bash scripts/inference_detection.sh
```

### 2. Point Cloud Feature Extraction
Extract re-identification features from detected objects:
```bash
bash scripts/inference_reid_extract.sh
```

### 3. Embedding Similarity Classification
Refine object classifications using similarity-based matching with known object galleries:
```bash
bash scripts/inference_reid_classification.sh
```

This step:
- Uses pre-trained galleries for FourierGR1T2 and AgilityDigit objects
- Calculates feature similarities between detected objects and gallery embeddings
- Refines classifications by reassigning Person (class 0) detections to FourierGR1T2 (class 4) or AgilityDigit (class 5) based on feature similarity
- Outputs refined detection results with improved object type classification

### 4. Short-Term Tracking
Perform single-camera tracking using 3D DeepSort (processes all 4 chunks automatically):
```bash
bash scripts/inference_short_term_tracking.sh
```

### 5. Long-Term Tracking
Perform trajectory re-association across time gaps (processes all chunks 0-3 automatically):
```bash
bash scripts/inference_long_term_tracking.sh
```

This step reconnects fragmented trajectories by:
- Analyzing trajectory endpoints and starts within time windows
- Using spatial distance and ReID feature similarity for trajectory matching
- Filtering trajectories based on minimum length requirements
- Outputs individual chunk files and a combined final result

### Switching Between Test and Validation
To process validation data instead of test data, modify the `split_name` parameter in the respective scripts:
- Edit `scripts/inference_detection.sh`: Change `--split_name "test"` to `--split_name "val"`
- Edit `scripts/inference_reid_extract.sh`: Change `--split_name "test"` to `--split_name "val"`  
- Edit `scripts/inference_reid_classification.sh`: Change `--split_name "test"` to `--split_name "val"`
- Edit `scripts/inference_short_term_tracking.sh`: Change `--split_name "test"` to `--split_name "val"`
- Edit `scripts/inference_long_term_tracking.sh`: Change `--split_name "test"` to `--split_name "val"`

## Evaluation

We use an adapted version of [TrackEval](https://github.com/JonathonLuiten/TrackEval) that supports 3D HOTA metrics for 3D bounding boxes.

### Run Evaluation
```bash
cd TrackEval
python main.py
```

The evaluation script uses the custom `AICity3D` dataset class:
```python
from trackeval.datasets.aicity_3d import AICity3D
```

This evaluates the final tracking results and provides 3D HOTA metrics suitable for the AI City Challenge 2025 Track1 evaluation.
