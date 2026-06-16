import torch
import torch.nn as nn
import numpy as np

def evidential_loss(mu, v, alpha, beta, y, is_ood=None, lam=0.2, ood_weight=5.0):
    two_beta_lamb = 2 * beta * (1 + v)
    loss_nll = (0.5 * torch.log(np.pi / v)
                - alpha * torch.log(two_beta_lamb)
                + (alpha + 0.5) * torch.log(v * (y - mu)**2 + two_beta_lamb)
                + torch.lgamma(alpha)
                - torch.lgamma(alpha + 0.5))
    loss_reg = torch.abs(y - mu) * (2 * v + alpha)
    loss = loss_nll + lam * loss_reg

    if is_ood is not None and is_ood.any():
        ep_var = beta[is_ood] / (v[is_ood] * (alpha[is_ood] - 1).clamp(min=1e-4))
        ood_penalty = ood_weight * torch.exp(-ep_var).mean()
        loss = torch.cat([loss[~is_ood], loss[is_ood] + ood_penalty.unsqueeze(0).expand(is_ood.sum(), loss.shape[-1])], dim=0)

    return loss.mean()


class EvidentialMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head    = nn.Linear(hidden, out_dim)
        self.v_head     = nn.Linear(hidden, out_dim)
        self.alpha_head = nn.Linear(hidden, out_dim)
        self.beta_head  = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h     = self.net(x)
        mu    = self.mu_head(h)
        v     = nn.functional.softplus(self.v_head(h)) + 1e-4
        alpha = nn.functional.softplus(self.alpha_head(h)) + 1 + 1e-4
        beta  = nn.functional.softplus(self.beta_head(h)) + 1e-4
        return mu, v, alpha, beta