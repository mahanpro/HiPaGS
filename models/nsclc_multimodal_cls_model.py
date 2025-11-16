import torch
import torch.nn as nn

from encoders.nsclc_image_encoder import ImageEncoder3D
from encoders.nsclc_ehr_encoder import EhrMLP
from heads.classification_head import BinaryClsHead


class NsclcMultimodalClsModel(nn.Module):
    def __init__(
        self, ehr_input_dim: int, img_emb_dim: int = 512, ehr_emb_dim: int = 128
    ):
        super().__init__()
        self.img_encoder = ImageEncoder3D(out_dim=img_emb_dim)
        self.ehr_encoder = EhrMLP(input_dim=ehr_input_dim, hidden=ehr_emb_dim)
        self.cls_head = BinaryClsHead(in_dim=img_emb_dim + ehr_emb_dim)

    def forward(self, image, ehr_feats):
        z_img = self.img_encoder(image)  # [B, img_emb_dim]
        z_ehr = self.ehr_encoder(ehr_feats)  # [B, ehr_emb_dim]
        z = torch.cat([z_img, z_ehr], dim=1)  # [B, img+ehr]
        logits = self.cls_head(z)  # [B]
        return logits, z
