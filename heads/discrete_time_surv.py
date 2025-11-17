import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiscreteTimeSurvHead(nn.Module):
    """
    Map fused patient vector -> per-bin hazards.

    Input:  x [N, in_dim]
    Output:
        logits   [N, n_bins]
        hazards  [N, n_bins] in (0,1)
        survival [N, n_bins] S(t_bin_k) = prod_{j<=k} (1 - h_ij)
    """

    def __init__(self, in_dim: int, n_bins: int):
        super().__init__()
        self.in_dim = in_dim
        self.n_bins = n_bins
        self.fc = nn.Linear(in_dim, n_bins)

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)  # [N, K]
        hazards = torch.sigmoid(logits)  # [N, K]
        one_minus = 1.0 - hazards
        survival = torch.cumprod(one_minus, dim=1)  # [N, K]
        return logits, hazards, survival


def make_time_bins(times: np.ndarray, n_bins: int = 12):
    """
    Build bin edges from continuous times (days) using quantiles.

    Returns:
        edges: np.ndarray of shape [n_bins], edges[k] is upper bound of bin k (0-based).
               Last bin is open-ended [edges[-1], +inf).
    """
    times = np.asarray(times, dtype=float)
    pos = times[np.isfinite(times) & (times > 0)]
    if pos.size == 0:
        raise ValueError("No positive finite times to build bins.")

    # quantiles from min to max
    qs = np.linspace(0.0, 1.0, n_bins, endpoint=True)
    edges = np.quantile(pos, qs)
    edges[0] = max(1.0, edges[0])  # avoid bin 0 being exactly 0
    return edges.astype(float)


def discretize_times_to_bins(
    times: torch.Tensor, bin_edges: torch.Tensor
) -> torch.Tensor:
    """
    Map continuous times to bin indices in [0, K-1].

    times: [N] (days)
    bin_edges: [K] ascending (upper bounds of each bin)

    For each time t:
      b = smallest k s.t. t <= bin_edges[k], else K-1.
    """
    if times.dim() != 1:
        raise ValueError("times must be 1D tensor")
    K = bin_edges.shape[0]
    # [N, K]
    t = times.unsqueeze(1)  # [N,1]
    edges = bin_edges.view(1, -1)  # [1,K]
    # boolean mask where t <= edge_k
    mask = t <= edges  # [N, K]
    # index of first True along dim=1; if none, set to K-1
    # Convert boolean to int by argmax over reversed mask trick
    # Use large value mask to compute index
    idx = mask.float()
    # if all False, row becomes zeros; we handle later
    first_true = torch.argmax(idx, dim=1)  # might be 0 even if mask all False
    # For rows where mask all False, set bin = K-1
    any_true = mask.any(dim=1)
    first_true = torch.where(any_true, first_true, torch.full_like(first_true, K - 1))
    return first_true.long()  # [N]


def discrete_time_nll(
    logits: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    bin_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood for discrete time hazards with censoring.

    logits:   [N, K]
    durations:[N] (days)
    events:   [N] in {0,1}
    bin_edges:[K] bin upper bounds

    For each patient i:
      b_i = bin index from durations[i]
      If event[i]==1:
        loglik_i = sum_{j< b_i} log(1-h_ij) + log(h_i,b_i)
      Else (censored):
        loglik_i = sum_{j<=b_i} log(1-h_ij)

    Returns scalar loss (mean over valid samples).
    """
    hazards = torch.sigmoid(logits)
    one_minus = 1.0 - hazards
    eps = 1e-8
    log_one_minus = torch.log(one_minus + eps)
    log_h = torch.log(hazards + eps)

    durations = durations.float()
    events = events.float()
    bin_edges = bin_edges.to(durations.device).float()

    # valid samples: positive finite durations
    valid = torch.isfinite(durations) & (durations > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    durations_v = durations[valid]
    events_v = events[valid]
    log_one_minus_v = log_one_minus[valid]
    log_h_v = log_h[valid]

    b = discretize_times_to_bins(durations_v, bin_edges)  # [Nv]
    Nv, K = log_h_v.shape

    # cumulative sum of log(1-h) along bins: [Nv, K]
    cum_log_surv = torch.cumsum(log_one_minus_v, dim=1)

    # gather helper
    idx_b = b.clamp(max=K - 1).unsqueeze(1)  # [Nv,1]

    # log survival up to bin b (inclusive)
    log_surv_to_b = cum_log_surv.gather(1, idx_b).squeeze(1)  # [Nv]

    # log survival up to bin b-1 (exclusive)
    b_minus_1 = (b - 1).clamp(min=0).unsqueeze(1)
    log_surv_to_bm1 = cum_log_surv.gather(1, b_minus_1).squeeze(1)
    # For true b = 0, survival up to b-1 is 0 (no bins)
    log_surv_to_bm1 = torch.where(
        b == 0, torch.zeros_like(log_surv_to_bm1), log_surv_to_bm1
    )

    # log hazard at bin b
    log_h_b = log_h_v.gather(1, idx_b).squeeze(1)

    is_event = events_v == 1.0
    is_cens = ~is_event

    # event loglik: survive until just before bin b, then fail in b
    loglik_event = log_surv_to_bm1 + log_h_b
    # censor loglik: survive through bin b
    loglik_cens = log_surv_to_b

    loglik = torch.zeros_like(loglik_event)
    loglik = torch.where(is_event, loglik_event, loglik)
    loglik = torch.where(is_cens, loglik_cens, loglik)

    # negative mean loglik over valid samples
    nll = -loglik.mean()
    return nll


def cumulative_hazard_from_hazards(hazards: torch.Tensor) -> torch.Tensor:
    """
    hazards: [N, K]
    Returns:
      cum_hazard: [N, K] where each entry is sum_{j<=k} -log(1-h_ij)
    """
    eps = 1e-8
    one_minus = 1.0 - hazards
    log_surv = torch.log(one_minus + eps)  # log S increments
    cum_log_surv = torch.cumsum(log_surv, dim=1)  # log S at each bin
    cum_hazard = -cum_log_surv
    return cum_hazard


def ranking_loss_from_hazards(
    hazards: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Pairwise ranking loss like DeepHit.

    hazards:  [N, K]
    durations:[N]
    events:   [N] in {0,1}
    alpha:    scale for loss, but we usually multiply this outside.

    Risk score per patient = sum_k cum_hazard_k (monotonic in hazard).
    Comparable pairs: i,j with duration_i < duration_j and event_i == 1.

    Loss = mean( log(1 + exp(-(r_i - r_j))) ) over comparable pairs.
    """
    durations = durations.float()
    events = events.float()

    valid = torch.isfinite(durations) & (durations > 0)
    if valid.sum() < 2:
        return torch.tensor(0.0, device=hazards.device, requires_grad=True)

    durations_v = durations[valid]
    events_v = events[valid]
    hazards_v = hazards[valid]

    # risk score from cumulative hazard
    cum_h = cumulative_hazard_from_hazards(hazards_v)  # [Nv, K]
    risk = cum_h.sum(dim=1)  # [Nv]

    Nv = risk.shape[0]

    # pairwise matrices
    t_i = durations_v.view(-1, 1)  # [Nv,1]
    t_j = durations_v.view(1, -1)  # [1,Nv]
    e_i = events_v.view(-1, 1)  # [Nv,1]

    # i before j and i had event
    comparable = (t_i < t_j) & (e_i == 1.0)

    if not comparable.any():
        return torch.tensor(0.0, device=hazards.device, requires_grad=True)

    r_i = risk.view(-1, 1)  # [Nv,1]
    r_j = risk.view(1, -1)  # [1,Nv]
    diff = r_i - r_j  # [Nv,Nv]

    # logistic pairwise loss
    pair_loss = F.softplus(-diff)  # log(1 + exp(-diff))
    loss = (pair_loss * comparable.float()).sum() / comparable.float().sum()

    return alpha * loss


def c_index(risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> float:
    """
    Concordance index (higher risk = shorter survival).

    risk:      [N] higher = worse
    durations: [N]
    events:    [N] in {0,1}
    """
    risk = risk.detach().cpu().numpy()
    t = durations.detach().cpu().numpy().astype(float)
    e = events.detach().cpu().numpy().astype(int)

    valid = np.isfinite(t) & (t > 0)
    risk = risk[valid]
    t = t[valid]
    e = e[valid]

    N = len(t)
    if N < 2:
        return float("nan")

    num = 0.0
    den = 0.0
    for i in range(N):
        if e[i] != 1:
            continue
        for j in range(N):
            if t[i] >= t[j]:
                continue
            den += 1.0
            if risk[i] > risk[j]:
                num += 1.0
            elif risk[i] == risk[j]:
                num += 0.5
    if den == 0:
        return float("nan")
    return float(num / den)


def km_censoring_weights(durations: np.ndarray, events: np.ndarray, t_grid: np.ndarray):
    """
    Kaplan Meier for censoring distribution G(t).

    durations: event or censor times
    events:    1 if event (death), 0 if censored
    t_grid:    times where we want G(t)

    We fit KM on censoring times where censoring is treated as event.

    Returns:
        G_grid: [M] estimate of P(C > t_grid[m])
    """
    # reverse roles: censor events (1 - events)
    d = np.asarray(durations, float)
    c = 1 - np.asarray(events, int)

    valid = np.isfinite(d) & (d > 0)
    d = d[valid]
    c = c[valid]

    order = np.argsort(d)
    times = d[order]
    cens_events = c[order]

    uniq_times = np.unique(times)
    n = len(times)
    G = 1.0
    G_vals = []
    t_vals = []

    idx = 0
    at_risk = n
    for t_u in uniq_times:
        # all indices at this time
        mask = times == t_u
        d_cens = cens_events[mask].sum()  # censoring "events"
        if at_risk > 0:
            G *= 1.0 - d_cens / at_risk
        G_vals.append(G)
        t_vals.append(t_u)
        at_risk -= mask.sum()

    t_vals = np.asarray(t_vals)
    G_vals = np.asarray(G_vals)

    G_grid = np.ones_like(t_grid, dtype=float)
    for i, t in enumerate(t_grid):
        # find last t_vals <= t
        mask = t_vals <= t
        if mask.any():
            G_grid[i] = G_vals[mask].max()
        else:
            G_grid[i] = 1.0
    return G_grid


def ibs_ipcw(
    survival: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    bin_edges: torch.Tensor,
) -> float:
    """
    Integrated Brier Score with IPCW over bin centers.

    survival: [N, K], S(t_bin_k)
    durations:[N]
    events:   [N]
    bin_edges:[K]

    Uses simple IPCW with censoring KM.
    """
    S = survival.detach().cpu().numpy()
    t = durations.detach().cpu().numpy().astype(float)
    e = events.detach().cpu().numpy().astype(int)
    edges = bin_edges.detach().cpu().numpy().astype(float)

    valid = np.isfinite(t) & (t > 0)
    S = S[valid]
    t = t[valid]
    e = e[valid]

    K = S.shape[1]
    if K == 0 or S.shape[0] == 0:
        return float("nan")

    # evaluation times: bin edges
    t_grid = edges
    G_grid = km_censoring_weights(t, e, t_grid)  # [K]

    brier_vals = []
    for k in range(K):
        tau = t_grid[k]
        S_tau = S[:, k]  # [N]
        # weights
        w = np.zeros_like(t, dtype=float)
        # cases with event before or at tau
        mask_event = (t <= tau) & (e == 1)
        # cases at risk at tau (t > tau)
        mask_risk = t > tau

        G_tau = max(G_grid[k], 1e-6)
        # G(t-) ~ G(t) for events
        G_tminus = np.zeros_like(t, dtype=float)
        G_tminus[mask_event] = G_tau
        G_tminus[mask_risk] = G_tau

        # event contribution: (0 - S(t))^2
        # risk contribution:  (1 - S(t))^2
        numer = 0.0
        denom = 0.0

        # events
        if mask_event.any():
            numer += np.sum(
                ((0.0 - S_tau[mask_event]) ** 2)
                / np.clip(G_tminus[mask_event], 1e-6, None)
            )
            denom += np.sum(1.0 / np.clip(G_tminus[mask_event], 1e-6, None))

        # at risk
        if mask_risk.any():
            numer += np.sum(
                ((1.0 - S_tau[mask_risk]) ** 2)
                / np.clip(G_tminus[mask_risk], 1e-6, None)
            )
            denom += np.sum(1.0 / np.clip(G_tminus[mask_risk], 1e-6, None))

        if denom > 0:
            brier_vals.append(numer / denom)

    if not brier_vals:
        return float("nan")

    # simple average over grid (approximate integral)
    return float(np.mean(brier_vals))
