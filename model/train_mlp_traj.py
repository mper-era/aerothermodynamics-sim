import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
from evidential_mlp import EvidentialMLP, evidential_loss

ROOT = Path(__file__).parent.parent
df = pd.read_parquet(ROOT / 'data' / 'trajectories.parquet')

# Match RSSM's trajectory-level train/val split so RMSE comparisons are apples-to-apples
np.random.seed(42)
traj_ids = df['traj_id'].unique()
perm = np.random.permutation(traj_ids)
n_val = int(0.15 * len(perm))
val_ids, train_ids = perm[:n_val], perm[n_val:]

train_df = df[df['traj_id'].isin(train_ids)]
val_df   = df[df['traj_id'].isin(val_ids)]

FEATURE_COLS = ['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']
TARGET_COLS  = ['CD', 'CL', 'q']

X_mean, X_std = train_df[FEATURE_COLS].mean().values, train_df[FEATURE_COLS].std().values
Y_mean, Y_std = train_df[TARGET_COLS].mean().values, train_df[TARGET_COLS].std().values

def norm(d, cols, mean, std):
    return torch.tensor(((d[cols].values - mean) / std).astype(np.float32))

X_train, Y_train = norm(train_df, FEATURE_COLS, X_mean, X_std), norm(train_df, TARGET_COLS, Y_mean, Y_std)
X_val, Y_val     = norm(val_df, FEATURE_COLS, X_mean, X_std), norm(val_df, TARGET_COLS, Y_mean, Y_std)

model = EvidentialMLP(in_dim=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

best_val, best_state = float('inf'), None
for epoch in range(300):
    model.train()
    mu, v, alpha, beta = model(X_train)
    loss = evidential_loss(mu, v, alpha, beta, Y_train)
    opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        mu_v, v_v, alpha_v, beta_v = model(X_val)
        val_loss = evidential_loss(mu_v, v_v, alpha_v, beta_v, Y_val).item()
    if val_loss < best_val:
        best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | val loss {val_loss:.4f}")

torch.save({'model': best_state, 'X_mean': X_mean, 'X_std': X_std, 'Y_mean': Y_mean, 'Y_std': Y_std},
           ROOT / 'model' / 'checkpoint_traj.pt')
print(f"Done. Best val loss: {best_val:.4f}")