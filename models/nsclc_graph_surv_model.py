"This is the full graph-aware survival model"

import torch
import torch.nn as nn

from gnn.graphsage import GraphSAGENet
from heads.discrete_time_surv import (
    DiscreteTimeSurvHead,
    cumulative_hazard_from_hazards,
)


class NsclcGraphSurvModel(nn.Module):
    """
    Graph-based survival model on top of fused patient features.

    Inputs to forward:
      e         [N, 512]   patient graph embeddings from PatientFeatureFusion
      z_img     [N, 512]
      z_ehr     [N, 128]
      z_concept [N, 64]
      edge_index [2, E]    from kNN

    Output dict:
      'logits'   [N, T]
      'hazards'  [N, T]
      'survival' [N, T]
      'h_graph'  [N, 256]  node embeddings from GNN
      'risk'     [N]       scalar risk score for C-index
    """

    def __init__(self, n_time_bins: int = 12):
        super().__init__()
        self.gnn = GraphSAGENet(in_dim=512, hidden_dim=256, num_layers=2, dropout=0.1)
        # fusion: [h_i (256) || z_img (512) || z_ehr (128) || z_concept (64)] = 960
        self.fuse = nn.Sequential(
            nn.Linear(256 + 512 + 128 + 64, 256),
            nn.ReLU(inplace=True),
        )
        self.surv_head = DiscreteTimeSurvHead(in_dim=256, n_bins=n_time_bins)
        self.n_time_bins = n_time_bins

    def forward(
        self,
        e: torch.Tensor,
        z_img: torch.Tensor,
        z_ehr: torch.Tensor,
        z_concept: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        # 1) GNN over graph using e as node feature
        h_graph = self.gnn(e, edge_index)  # [N, 256]

        # 2) Fuse with original modalities
        fused = torch.cat([h_graph, z_img, z_ehr, z_concept], dim=1)  # [N, 960]
        fused = self.fuse(fused)  # [N, 256]

        # 3) Survival head
        logits, hazards, survival = self.surv_head(fused)  # [N,T] each

        # risk score from cumulative hazard
        cum_h = cumulative_hazard_from_hazards(hazards)  # [N,T]
        risk = cum_h.sum(dim=1)  # [N]

        return {
            "logits": logits,
            "hazards": hazards,
            "survival": survival,
            "h_graph": h_graph,
            "risk": risk,
        }
