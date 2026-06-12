import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, out_dim * 4)
        self.out_dim = out_dim

    def forward(self, x):
        h   = self.backbone(x)
        out = self.head(h).reshape(-1, self.out_dim, 4)
        mu    = out[..., 0]
        v     = F.softplus(out[..., 1]) + 1e-4
        alpha = F.softplus(out[..., 2]) + 1.0
        beta  = F.softplus(out[..., 3]) + 1e-4
        return mu, v, alpha, beta

def evidential_loss(mu, v, alpha, beta, y, lam=0.2):
    omega = 2 * beta * (1 + v)
    nll   = (0.5 * torch.log(torch.pi / v)
             - alpha * torch.log(omega)
             + (alpha + 0.5) * torch.log(v * (y - mu)**2 + omega)
             + torch.lgamma(alpha)
             - torch.lgamma(alpha + 0.5))
    reg = torch.abs(y - mu) * (2 * v + alpha)
    return (nll + lam * reg).mean()
