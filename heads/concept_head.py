"""
This takes a base embedding base_emb = [z_img ∥ z_ehr] of size 512 + 128 = 640 and predicts:

- 6 binary concepts (logits shape [B])
- 2 ordinal concepts (e.g. 3 bins; logits shape [B, num_bins])
- projected concept embeddings (64-D total)
- the mask vector m from the concept metadata (not learned)
"""

import torch
import torch.nn as nn
from typing import Dict, List


BINARY_CONCEPTS = ["EGFR", "KRAS", "ALK", "Adjuvant", "Chemo", "Radiation"]
ORDINAL_CONCEPTS = ["TumVolBin", "SUVPeakBin"]


class ConceptHead(nn.Module):
    """
    Concept head on top of base embedding [z_img ∥ z_ehr].

    - Input: base_emb [B, base_dim]
    - Output:
        - binary_logits: dict[name] -> [B]
        - ordinal_logits: dict[name] -> [B, n_classes]
        - z_concept: [B, proj_dim * num_concepts] (e.g. 64)
        - mask: [B, num_concepts] from provided concept_missing
    """

    def __init__(
        self,
        base_dim: int = 512 + 128,
        proj_dim: int = 8,
        tumvol_bins: int = 3,
        suvpeak_bins: int = 3,
    ):
        super().__init__()

        self.base_dim = base_dim
        self.proj_dim = proj_dim

        # Binary heads: one logit per concept
        self.binary_heads = nn.ModuleDict(
            {name: nn.Linear(base_dim, 1) for name in BINARY_CONCEPTS}
        )
        self.binary_proj = nn.ModuleDict(
            {name: nn.Linear(1, proj_dim) for name in BINARY_CONCEPTS}
        )

        # Ordinal heads: multi-class logits per concept
        self.ordinal_heads = nn.ModuleDict(
            {
                "TumVolBin": nn.Linear(base_dim, tumvol_bins),
                "SUVPeakBin": nn.Linear(base_dim, suvpeak_bins),
            }
        )
        self.ordinal_proj = nn.ModuleDict(
            {
                "TumVolBin": nn.Linear(tumvol_bins, proj_dim),
                "SUVPeakBin": nn.Linear(suvpeak_bins, proj_dim),
            }
        )

        self.num_concepts = len(BINARY_CONCEPTS) + len(ORDINAL_CONCEPTS)
        self.out_dim = self.num_concepts * proj_dim  # e.g. 8 * 8 = 64

    def forward(
        self,
        base_emb: torch.Tensor,
        concept_missing: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        """
        base_emb: [B, base_dim] = [z_img ∥ z_ehr]
        concept_missing: [B, num_concepts] in {0,1}, order:
            [EGFR, KRAS, ALK, Adjuvant, Chemo, Radiation, TumVolBin, SUVPeakBin]

        If concept_missing is None, mask = zeros.
        """
        B = base_emb.size(0)
        device = base_emb.device

        # 1) Binary logits
        binary_logits: Dict[str, torch.Tensor] = {}
        binary_proj_vecs: List[torch.Tensor] = []

        for name in BINARY_CONCEPTS:
            logit = self.binary_heads[name](base_emb).squeeze(-1)  # [B]
            binary_logits[name] = logit
            # project [B,1] -> [B,proj_dim]
            proj_vec = self.binary_proj[name](logit.unsqueeze(-1))
            binary_proj_vecs.append(proj_vec)

        # 2) Ordinal logits
        ordinal_logits: Dict[str, torch.Tensor] = {}
        ordinal_proj_vecs: List[torch.Tensor] = []

        for name in ORDINAL_CONCEPTS:
            logits = self.ordinal_heads[name](base_emb)  # [B, n_classes]
            ordinal_logits[name] = logits
            proj_vec = self.ordinal_proj[name](logits)  # [B, proj_dim]
            ordinal_proj_vecs.append(proj_vec)

        # 3) Concatenate all concept projections -> z_concept
        all_proj = binary_proj_vecs + ordinal_proj_vecs
        z_concept = torch.cat(all_proj, dim=1)  # [B, num_concepts * proj_dim]

        # 4) Mask vector m
        if concept_missing is None:
            mask = torch.zeros(
                (B, self.num_concepts), dtype=torch.float32, device=device
            )
        else:
            if concept_missing.shape != (B, self.num_concepts):
                raise ValueError(
                    f"concept_missing shape {concept_missing.shape} "
                    f"does not match (B, {self.num_concepts})"
                )
            mask = concept_missing.to(device=device, dtype=torch.float32)

        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "z_concept": z_concept,
            "mask": mask,
        }
