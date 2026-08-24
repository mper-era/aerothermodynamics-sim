import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
from evidential_mlp import EvidentialMLP, evidential_loss

ROOT = Path(__file__).parent.parent
FEATURE_COLS = ['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']
TARGET_COLS = ['CL_mean', 'CD_mean', 'q_mean']  # matches original train.py's order

# --- Load base data, fixed split (identical seed/logic to train.py) ---
df = pd.read_parquet(ROOT / 'data' / 'processed.parquet')
np.random.seed(42)
idx = np.random.permutation(len(df))
n_val = int(0.15 * len(df))
val_idx, train_idx = idx[:n_val], idx[n_val:]

val_df = df.iloc[val_idx].reset_index(drop=True)
base_train_df = df.iloc[train_idx].reset_index(drop=True)

# --- Load active learning results, rename to match TARGET_COLS convention ---
def load_al_csv(path):
    d = pd.read_csv(path)
    d = d.rename(columns={'CD': 'CD_mean', 'CL': 'CL_mean', 'q': 'q_mean'})
    return d[FEATURE_COLS + TARGET_COLS]

uncertainty_df = load_al_csv(ROOT / 'data' / 'active_learning_uncertainty.csv')
random_df = load_al_csv(ROOT / 'data' / 'active_learning_random.csv')

train_sets = {
    'baseline (no new points)': base_train_df[FEATURE_COLS + TARGET_COLS],
    '+15 uncertainty-guided': pd.concat([base_train_df[FEATURE_COLS + TARGET_COLS], uncertainty_df], ignore_index=True),
    '+15 random': pd.concat([base_train_df[FEATURE_COLS + TARGET_COLS], random_df], ignore_index=True),
}

def train_and_eval(train_df, label, epochs=300, seed=0):
    torch.manual_seed(seed)
    X_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    Y_raw = train_df[TARGET_COLS].values.astype(np.float32)
    X_mean, X_std = X_raw.mean(0), X_raw.std(0)
    Y_mean, Y_std = Y_raw.mean(0), Y_raw.std(0)

    X_train = torch.tensor((X_raw - X_mean) / X_std)
    Y_train = torch.tensor((Y_raw - Y_mean) / Y_std)

    model = EvidentialMLP(in_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        mu, v, alpha, beta = model(X_train)
        loss = evidential_loss(mu, v, alpha, beta, Y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    X_val = torch.tensor(((val_df[FEATURE_COLS].values.astype(np.float32) - X_mean) / X_std))
    Y_val_true = val_df[TARGET_COLS].values
    with torch.no_grad():
        mu_val, _, _, _ = model(X_val)
    Y_val_pred = mu_val.numpy() * Y_std + Y_mean

    rmse = np.sqrt(((Y_val_pred - Y_val_true) ** 2).mean(axis=0))
    print(f"{label:<28} n_train={len(train_df):>4}  "
          f"CL_rmse={rmse[0]:.4f}  CD_rmse={rmse[1]:.4f}  q_rmse={rmse[2]:.2f}")
    return rmse

print(f"{'Strategy':<28} {'n_train':<10} {'RMSE (CL / CD / q)'}")
print("-" * 80)
results = {}
for label, tdf in train_sets.items():
    results[label] = train_and_eval(tdf, label)