import torch.nn as nn

from encoders.nsclc_image_encoder import ImageEncoder3D
from heads.survival_head import CoxHead


class NsclcImageSurvModel(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.encoder = ImageEncoder3D(out_dim=emb_dim)
        self.head = CoxHead(in_dim=emb_dim)

    def forward(self, image):
        z = self.encoder(image)  # [B, emb_dim]
        risk = self.head(z)  # [B]
        return risk, z
