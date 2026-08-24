import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
from evidential_mlp import EvidentialMLP, evidential_loss

ROOT = Path(__file__).parent.parent
df = pd.read_parquet(ROOT / 'data' / 'processed.parquet')
FEATURE_COLS = ['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']
TARGET_COLS = ['CL_mean', 'CD_mean', 'q_mean']

N_SEEDS = 15

def run_seed(seed, epochs=300):
    np.random.seed(seed)
    torch.manual_seed(seed)
    idx = np.random.permutation(len(df))
    n_val = int(0.15 * len(df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    Y_raw = df[TARGET_COLS].values.astype(np.float32)
    X_mean, X_std = X_raw[train_idx].mean(0), X_raw[train_idx].std(0)
    Y_mean, Y_std = Y_raw[train_idx].mean(0), Y_raw[train_idx].std(0)

    X_train = torch.tensor((X_raw[train_idx] - X_mean) / X_std)
    Y_train = torch.tensor((Y_raw[train_idx] - Y_mean) / Y_std)
    X_val = torch.tensor((X_raw[val_idx] - X_mean) / X_std)
    Y_val_true = Y_raw[val_idx]

    model = EvidentialMLP(in_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        model.train()
        mu, v, alpha, beta = model(X_train)
        loss = evidential_loss(mu, v, alpha, beta, Y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        mu_val, _, _, _ = model(X_val)
    Y_val_pred = mu_val.numpy() * Y_std + Y_mean
    rmse = np.sqrt(((Y_val_pred - Y_val_true) ** 2).mean(axis=0))
    print(f"  seed {seed} done: CL={rmse[0]:.4f} CD={rmse[1]:.4f} q={rmse[2]:,.2f}")
    return rmse

print(f"Running {N_SEEDS} seeds...")
results = np.array([run_seed(s) for s in range(N_SEEDS)])

print(f"\n{'seed':<6}{'CL_rmse':>12}{'CD_rmse':>12}{'q_rmse':>18}")
for s in range(N_SEEDS):
    print(f"{s:<6}{results[s,0]:>12.4f}{results[s,1]:>12.4f}{results[s,2]:>18,.2f}")

mean_r, std_r = results.mean(axis=0), results.std(axis=0)
print(f"\n{'mean':<6}{mean_r[0]:>12.4f}{mean_r[1]:>12.4f}{mean_r[2]:>18,.2f}")
print(f"{'std':<6}{std_r[0]:>12.4f}{std_r[1]:>12.4f}{std_r[2]:>18,.2f}")

cv = std_r / mean_r * 100
print(f"\nCoefficient of variation (std/mean):")
print(f"  CL: {cv[0]:.1f}%   CD: {cv[1]:.1f}%   q: {cv[2]:.1f}%")

np.save(ROOT / 'data' / 'seed_sweep_results.npy', results)

results_df = pd.DataFrame(results, columns=['CL_rmse', 'CD_rmse', 'q_rmse'])
results_df.index.name = 'seed'
results_df.to_csv(ROOT / 'data' / 'seed_sweep_results.csv')

print(f"Saved raw per-seed results to data/seed_sweep_results.npy and data/seed_sweep_results.csv")