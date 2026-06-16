import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from evidential_mlp import EvidentialMLP, evidential_loss

FEATURE_COLS = ['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']
TARGET_COLS  = ['CL_mean', 'CD_mean', 'q_mean']
EPOCHS       = 300
BATCH        = 32
LR           = 3e-4

# Load data
df = pd.read_parquet('../data/processed.parquet')
X_raw = df[FEATURE_COLS].values.astype(np.float32)
Y_raw = df[TARGET_COLS].values.astype(np.float32)

X_mean, X_std = X_raw.mean(0), X_raw.std(0)
Y_mean, Y_std = Y_raw.mean(0), Y_raw.std(0)
X_norm = (X_raw - X_mean) / X_std
Y_norm = (Y_raw - Y_mean) / Y_std

# Train/val split
idx     = np.random.permutation(len(X_norm))
n_val   = int(0.15 * len(idx))
val_idx, train_idx = idx[:n_val], idx[n_val:]
X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
X_val,   Y_val   = X_norm[val_idx],   Y_norm[val_idx]

# Augment training set with OOD points
import os
OOD_PATH = '../data/ood_processed.parquet'
if os.path.exists(OOD_PATH):
    ood_df    = pd.read_parquet(OOD_PATH)
    X_ood_raw = ood_df[FEATURE_COLS].values.astype(np.float32)
    Y_ood_raw = ood_df[TARGET_COLS].values.astype(np.float32)
    X_ood_norm = (X_ood_raw - X_mean) / X_std
    Y_ood_norm = (Y_ood_raw - Y_mean) / Y_std
    X_train    = np.concatenate([X_train, X_ood_norm])
    Y_train    = np.concatenate([Y_train, Y_ood_norm])
    is_ood     = np.array([False]*len(train_idx) + [True]*len(X_ood_norm))
    print(f'Augmented with {len(X_ood_norm)} OOD points')
else:
    is_ood = np.array([False]*len(X_train))
    print('No OOD data found, training without augmentation')

# Datasets
train_ds = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(Y_train),
    torch.tensor(is_ood, dtype=torch.bool)
)
val_ds = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(Y_val),
    torch.tensor(np.zeros(len(X_val), dtype=bool))
)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)

# Model
model     = EvidentialMLP(in_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val, best_state = float('inf'), None

for epoch in range(EPOCHS):
    model.train()
    for X_b, Y_b, ood_b in train_loader:
        optimizer.zero_grad()
        mu, v, alpha, beta = model(X_b)
        loss = evidential_loss(mu, v, alpha, beta, Y_b, is_ood=ood_b)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_b, Y_b, ood_b in val_loader:
            mu, v, alpha, beta = model(X_b)
            val_losses.append(evidential_loss(mu, v, alpha, beta, Y_b).item())
    val_loss = np.mean(val_losses)
    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 50 == 0:
        print(f'Epoch {epoch:3d} | val loss {val_loss:.4f}')

torch.save({
    'model':  best_state,
    'X_mean': X_mean, 'X_std': X_std,
    'Y_mean': Y_mean, 'Y_std': Y_std,
}, '../model/checkpoint.pt')
print(f'Done. Best val loss: {best_val:.4f}')