import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from rssm import SimplifiedRSSM
from evidential_mlp import evidential_loss

FEATURE_COLS = ['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']
TARGET_COLS  = ['CL', 'CD', 'q']
KL_WEIGHT    = 0.1

df = pd.read_parquet('../data/trajectories.parquet')

X_all = df[FEATURE_COLS].values.astype(np.float32)
Y_all = df[TARGET_COLS].values.astype(np.float32)
X_mean, X_std = X_all.mean(0), X_all.std(0)
Y_mean, Y_std = Y_all.mean(0), Y_all.std(0)

traj_ids = df['traj_id'].unique()
np.random.seed(42)
np.random.shuffle(traj_ids)
n_val     = int(0.15 * len(traj_ids))
val_ids   = set(traj_ids[:n_val])
train_ids = set(traj_ids[n_val:])

def make_tensors(ids):
    seqs_x, seqs_y = [], []
    for tid in ids:
        sub = df[df['traj_id'] == tid].sort_values('time_s')
        x = torch.tensor((sub[FEATURE_COLS].values - X_mean) / X_std,
                         dtype=torch.float32)
        y = torch.tensor((sub[TARGET_COLS].values  - Y_mean) / Y_std,
                         dtype=torch.float32)
        seqs_x.append(x)
        seqs_y.append(y)
    return seqs_x, seqs_y

def collate(batch):
    xs, ys = zip(*batch)
    return pad_sequence(xs, batch_first=True), pad_sequence(ys, batch_first=True)

train_x, train_y = make_tensors(train_ids)
val_x,   val_y   = make_tensors(val_ids)
train_dl = DataLoader(list(zip(train_x, train_y)), batch_size=16,
                      shuffle=True, collate_fn=collate)
val_dl   = DataLoader(list(zip(val_x,   val_y)),   batch_size=16,
                      collate_fn=collate)

model = SimplifiedRSSM(obs_dim=4, hidden_dim=64, latent_dim=32, out_dim=3)
opt   = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(300):
    model.train()
    train_losses = []
    for xb, yb in train_dl:
        mu, v, alpha, beta, kl = model(xb)
        recon = evidential_loss(
            mu.reshape(-1, 3),    v.reshape(-1, 3),
            alpha.reshape(-1, 3), beta.reshape(-1, 3),
            yb.reshape(-1, 3)
        )
        loss = recon + KL_WEIGHT * kl
        opt.zero_grad(); loss.backward(); opt.step()
        train_losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                mu, v, alpha, beta, kl = model(xb)
                recon = evidential_loss(
                    mu.reshape(-1, 3),    v.reshape(-1, 3),
                    alpha.reshape(-1, 3), beta.reshape(-1, 3),
                    yb.reshape(-1, 3)
                )
                val_losses.append((recon + KL_WEIGHT * kl).item())
        print(f'Epoch {epoch+1:3d} | train {np.mean(train_losses):.4f} | val {np.mean(val_losses):.4f}')

torch.save({
    'model':        model.state_dict(),
    'X_mean':       X_mean,  'X_std':  X_std,
    'Y_mean':       Y_mean,  'Y_std':  Y_std,
    'feature_cols': FEATURE_COLS,
    'target_cols':  TARGET_COLS,
}, 'rssm_checkpoint.pt')
print('Saved rssm_checkpoint.pt')
