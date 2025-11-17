"This is a baseline image-only Cox survival model"

import torch.nn as nn

from encoders.nsclc_image_encoder import NsclcImageEncoder
from heads.survival_head import CoxHead


class NsclcImageSurvModel(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.encoder = NsclcImageEncoder()
        self.head = CoxHead(in_dim=emb_dim)

    def forward(self, image):
        z = self.encoder(image)  # [B, emb_dim]
        risk = self.head(z)  # [B]
        return risk, z
