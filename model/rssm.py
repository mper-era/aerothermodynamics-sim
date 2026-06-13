import torch
import torch.nn as nn
import torch.nn.functional as F
from evidential_mlp import evidential_loss

class SimplifiedRSSM(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=64, latent_dim=32, out_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru = nn.GRUCell(input_size=obs_dim, hidden_size=hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim * 4)
        )
        self.out_dim = out_dim

    def prior(self, h):
        params = self.prior_net(h)
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, F.softplus(log_sigma) + 1e-4

    def posterior(self, h, o):
        params = self.posterior_net(torch.cat([h, o], dim=-1))
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, F.softplus(log_sigma) + 1e-4

    def decode(self, z, h):
        x   = torch.cat([z, h], dim=-1)
        out = self.decoder(x).reshape(*x.shape[:-1], self.out_dim, 4)
        mu    = out[..., 0]
        v     = F.softplus(out[..., 1]) + 1e-4
        alpha = F.softplus(out[..., 2]) + 1.0
        beta  = F.softplus(out[..., 3]) + 1e-4
        return mu, v, alpha, beta

    def forward(self, obs_seq):
        B, T, _ = obs_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        all_mu, all_v, all_alpha, all_beta, all_kl = [], [], [], [], []

        for t in range(T):
            o_t = obs_seq[:, t, :]
            prior_mu, prior_sigma = self.prior(h)
            post_mu,  post_sigma  = self.posterior(h, o_t)
            eps = torch.randn_like(post_mu)
            z_t = post_mu + eps * post_sigma

            kl = 0.5 * (
                (post_sigma / prior_sigma).pow(2)
                + ((prior_mu - post_mu) / prior_sigma).pow(2)
                - 1
                + 2 * (prior_sigma.log() - post_sigma.log())
            ).sum(dim=-1).mean()

            mu, v, alpha, beta = self.decode(z_t, h)
            h = self.gru(o_t, h)

            all_mu.append(mu)
            all_v.append(v)
            all_alpha.append(alpha)
            all_beta.append(beta)
            all_kl.append(kl)

        return (torch.stack(all_mu,    dim=1),
                torch.stack(all_v,     dim=1),
                torch.stack(all_alpha, dim=1),
                torch.stack(all_beta,  dim=1),
                torch.stack(all_kl,    dim=0).mean())

    def rollout(self, obs_seq):
        self.eval()
        with torch.no_grad():
            obs = obs_seq.unsqueeze(0)
            h = torch.zeros(1, self.hidden_dim)
            means, ep_stds = [], []

            for t in range(obs.shape[1]):
                o_t = obs[:, t, :]
                prior_mu, prior_sigma = self.prior(h)
                z_t = prior_mu + torch.randn_like(prior_mu) * prior_sigma
                mu, v, alpha, beta = self.decode(z_t, h)
                ep_var = beta / (v * (alpha - 1).clamp(min=1e-4))
                ep_std = ep_var.sqrt()
                means.append(mu.squeeze(0))
                ep_stds.append(ep_std.squeeze(0))
                h = self.gru(o_t, h)

            return (torch.stack(means,   dim=0),
                    torch.stack(ep_stds, dim=0))
