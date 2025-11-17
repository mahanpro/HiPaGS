import torch
from torch import nn
from monai.networks.nets.densenet import DenseNet121


class NsclcImageEncoder(nn.Module):
    """
    Simple 3D image encoder for PET/CT.

    Expects input: [B, 2, D, H, W]
    Returns:       [B, img_emb_dim]
    """

    def __init__(self, in_channels: int = 2, img_emb_dim: int = 512):
        super().__init__()
        # DenseNet121 already does global pooling + linear classifier
        # and returns [B, img_emb_dim].
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=img_emb_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, D, H, W]
        feat = self.backbone(x)  # usually [B, img_emb_dim]

        # In case you later swap to a conv backbone that returns [B, C, D, H, W],
        # handle both cases safely.
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)

        return feat
