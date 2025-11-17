import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """
    Simple GraphSAGE layer with mean aggregation and residual + LayerNorm.

    Conventions:
      edge_index: [2, E], edge_index[0] = src, edge_index[1] = dst
      Messages flow src -> dst, aggregation is mean over incoming neighbors.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin_agg = nn.Linear(2 * in_dim, out_dim)  # [x_i || mean(N(i))] -> out_dim
        self.res_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_dim]
        edge_index: [2, E]
        """
        N, D = x.shape
        assert D == self.in_dim

        src, dst = edge_index  # each [E]

        # Messages are just x[src]
        messages = x[src]  # [E, D]

        # Aggregate by mean over dst nodes
        agg = x.new_zeros((N, D))
        deg = x.new_zeros((N, 1))

        agg.index_add_(0, dst, messages)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype).unsqueeze(-1))

        deg = deg.clamp(min=1.0)
        agg = agg / deg  # [N, D]

        # Combine self and neighbors
        h_in = torch.cat([x, agg], dim=-1)  # [N, 2D]
        h = self.lin_agg(h_in)  # [N, out_dim]
        h = self.dropout(h)

        # Residual + LayerNorm
        res = self.res_proj(x)  # [N, out_dim]
        out = self.ln(h + res)
        out = F.relu(out)

        return out


class GraphSAGENet(nn.Module):
    """
    Stack of GraphSAGE layers.

    Example:
        net = GraphSAGENet(in_dim=512, hidden_dim=256, num_layers=2)
        h = net(e, edge_index)  # [N, 256]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for d_in in dims:
            layers.append(GraphSAGELayer(d_in, hidden_dim, dropout=dropout))

        self.layers = nn.ModuleList(layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x  # [N, hidden_dim]
