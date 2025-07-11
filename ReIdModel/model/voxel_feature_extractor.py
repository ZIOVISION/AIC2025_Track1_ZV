# Copyright (c) V-DETR authors. All Rights Reserved.
import torch
import torch.nn as nn
import MinkowskiEngine as ME

import torch.nn.functional as F


class CustomBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(kwargs.get('in_channels', 3), 64, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(),
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(256, 512, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.conv5 = nn.Sequential(
            ME.MinkowskiConvolution(512, 1024, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(1024),
            ME.MinkowskiReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return [x1,x2,x3,x4,x5]

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, in_channels=6, feature_dim=512, voxel_size=0.02, resnet_depth=34):
        super().__init__()
        self.voxel_size = voxel_size
        
        self.backbone = CustomBackbone(in_channels=in_channels, depth=resnet_depth)


        self.global_pool1 = ME.MinkowskiGlobalMaxPooling()
        self.global_pool2 = ME.MinkowskiGlobalMaxPooling()
        self.global_pool3 = ME.MinkowskiGlobalMaxPooling()
        self.global_pool4 = ME.MinkowskiGlobalMaxPooling()
        self.global_pool5 = ME.MinkowskiGlobalMaxPooling()

        combined_feature_dim = 1792+64 + 128
        self.feature_projection = nn.Linear(combined_feature_dim, feature_dim)

    def forward(self, point_clouds):
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p) for p in point_clouds],
            dtype=torch.float32, device=point_clouds[0].device
        )
        s_input = ME.SparseTensor(features=features, coordinates=coordinates)
        
        x1,x2,x3, x4, x5 = self.backbone(s_input)

        pooled_x1 = self.global_pool1(x1)
        pooled_x2 = self.global_pool2(x2)
        pooled_x3 = self.global_pool3(x3) # (B, 256)
        pooled_x4 = self.global_pool4(x4) # (B, 512)
        pooled_x5 = self.global_pool5(x5) # (B, 1024)

        combined_features = torch.cat([pooled_x1.F,pooled_x2.F,pooled_x3.F, pooled_x4.F, pooled_x5.F], dim=1)
        # combined_features shape: (B, 256 + 512 + 1024) = (B, 1792)

        projected_features = self.feature_projection(combined_features)
        
        normalized_features = F.normalize(projected_features, p=2, dim=1)
        
        return normalized_features