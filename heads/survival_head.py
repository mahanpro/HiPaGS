import torch
import torch.nn as nn


class CoxHead(nn.Module):
    """
    Maps embedding -> scalar risk score.
    Higher score = higher hazard.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # returns [B,]
        return self.fc(z).squeeze(-1)


def cox_ph_loss(
    risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    """
    Negative Cox partial log likelihood.
    risk_scores: [N], higher = worse
    times: [N], event or censor time
    events: [N] in {0,1}
    """
    # sort by time descending
    order = torch.argsort(times, descending=True)
    r = risk_scores[order]
    e = events[order]

    # cumulative log sum exp of risk
    # use cumsum in exp space for efficiency
    max_r = torch.max(r)
    exp_r = torch.exp(r - max_r)
    cum_sum = torch.cumsum(exp_r, dim=0)
    log_cum_sum = torch.log(cum_sum) + max_r

    # log partial likelihood
    # only for events
    # sum_e ( r_i - log sum_j>=i exp(r_j) )
    ll = (r - log_cum_sum) * e
    # negative mean
    return -ll.sum() / (e.sum() + 1e-8)
