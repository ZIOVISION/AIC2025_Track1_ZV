import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import glob
from collections import defaultdict
import random
import argparse

import configs.best as cfg

from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses, miners
import torch.nn.functional as F


from dataset import create_dataloader, create_query_gallery_split, ListBasedReIDDataset, read_ply_to_tensor
from model import VoxelFeatureExtractor

def extract_features(model, dataloader, device):
    model.eval()
    features_list, pids_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Validation Features"):
            point_clouds, pids = batch
            point_clouds = [pc.to(device) for pc in point_clouds]
            embeddings = model(point_clouds)
            features_list.append(embeddings.cpu())
            pids_list.append(pids)
    features = torch.cat(features_list, dim=0)
    pids = torch.cat(pids_list, dim=0).numpy()
    return features, pids

def evaluate(model, query_loader, gallery_loader, device):
    query_features, query_pids = extract_features(model, query_loader, device)
    gallery_features, gallery_pids = extract_features(model, gallery_loader, device)
    
    dist_matrix = torch.cdist(query_features, gallery_features, p=2).numpy()
    num_query = query_features.shape[0]
    
    indices = np.argsort(dist_matrix, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis])
    
    all_cmc, all_ap = [], []
    for i in range(num_query):
        cmc = matches[i].cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        if not np.any(matches[i]):
            all_ap.append(0.0)
            continue
        precision = np.arange(1, matches[i].sum() + 1) / (np.where(matches[i])[0] + 1)
        all_ap.append(np.mean(precision))

    cmc = np.asarray(all_cmc).astype(np.float32).sum(axis=0) / num_query
    rank1, rank5, mAP = cmc[0], cmc[4], np.mean(all_ap)
    return rank1, rank5, mAP


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReID model on point cloud data")
    parser.add_argument('--data_root', type=str, default='../dataset/obj_crop_pcd_dataset', 
                       help="Root directory of the ReID training dataset")
    parser.add_argument('--output_dir', type=str, default='./weights',
                       help="Directory to save trained model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(cfg.DEVICE)
    
    # Update paths based on arguments
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    output_dir = args.output_dir
    
    print(f"Training data directory: {train_dir}")
    print(f"Validation data directory: {val_dir}")
    print(f"Output directory: {output_dir}")
    
    print("Creating Training Dataloader...")
    train_loader, num_pids = create_dataloader(train_dir, use_augmentation=True)
    print(f"Training Dataloader created. Number of classes (PIDs): {num_pids}")
    
    query_list, gallery_list = create_query_gallery_split(val_dir)
    query_dataset = ListBasedReIDDataset(query_list)
    gallery_dataset = ListBasedReIDDataset(gallery_list)
    val_query_loader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=cfg.NUM_WORKERS)
    val_gallery_loader = DataLoader(gallery_dataset, batch_size=128, shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    print("Building model...")
    model = VoxelFeatureExtractor().to(device)

    print("Setting up loss, miner, and optimizer...")
    loss_func = losses.TripletMarginLoss(margin=cfg.LOSS_MARGIN)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_SCHEDULER_STEP, gamma=cfg.LR_SCHEDULER_GAMMA)

    print("Starting training...")
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            point_clouds, pids, _ = batch
            pids = pids.to(device)
            point_clouds_list = [pc.to(device) for pc in point_clouds]
            
            embeddings = model(point_clouds_list)
            hard_pairs = miner(embeddings, pids)
            loss = loss_func(embeddings, pids, hard_pairs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % cfg.LOG_PERIOD == 0:
                print(f"Epoch [{epoch}/{cfg.EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch [{epoch}/{cfg.EPOCHS}] Finished ---")
        print(f"Average Training Loss: {avg_loss:.4f}, Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if epoch % 10 == 0:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f"reid_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            rank1, rank5, mAP = evaluate(model, val_query_loader, val_gallery_loader, device)
            print("--- Validation Results ---")
            print(f"mAP: {mAP:.2%}")
            print(f"Rank-1 Accuracy: {rank1:.2%}")
            print(f"Rank-5 Accuracy: {rank5:.2%}")
            print("--------------------------\n")

if __name__ == "__main__":
    main()