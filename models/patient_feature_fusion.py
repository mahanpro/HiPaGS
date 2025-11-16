import torch
import torch.nn as nn
import torch.nn.functional as F


class PatientFeatureFusion(nn.Module):
    """
    Build patient graph embedding e_i from:
    - z_img [B, 512]
    - z_ehr [B, 128]
    - z_concept [B, 64]
    - mask [B, 8] (concept missing bits)

    Output:
    - e [B, 512] L2-normalized
    - concat_raw [B, 512+128+64+8] (optional for debugging)
    """

    def __init__(
        self, img_dim=512, ehr_dim=128, concept_dim=64, mask_dim=8, out_dim=512
    ):
        super().__init__()
        in_dim = img_dim + ehr_dim + concept_dim + mask_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
        self.out_dim = out_dim
        self.in_dim = in_dim

    def forward(
        self,
        z_img: torch.Tensor,
        z_ehr: torch.Tensor,
        z_concept: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        All inputs shape [B, *].
        """
        if (
            z_img.dim() != 2
            or z_ehr.dim() != 2
            or z_concept.dim() != 2
            or mask.dim() != 2
        ):
            raise ValueError("All inputs must be [B, D] tensors.")

        x = torch.cat([z_img, z_ehr, z_concept, mask], dim=1)  # [B, in_dim]
        h = self.proj(x)  # [B, out_dim]
        e = F.normalize(h, p=2, dim=1)  # L2-normalized

        return e, x
