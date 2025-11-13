import torch.nn as nn
from monai.networks.nets import resnet


class ImageEncoder3D(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.backbone = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=0,  # no final FC, backbone returns feature map
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(512, out_dim)  # ResNet18 final channels = 512
        self.out_dim = out_dim

    def forward(self, x):
        feat = self.backbone(x)  # [B, 512, d, h, w]
        pooled = self.pool(feat).view(x.size(0), -1)  # [B, 512]
        emb = self.proj(pooled)  # [B, out_dim]
        return emb
