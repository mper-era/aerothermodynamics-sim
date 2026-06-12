import sys
sys.path.insert(0, '.')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from evidential_mlp import EvidentialMLP, evidential_loss

df = pd.read_parquet('../data/processed.parquet').dropna()

X_raw = df[['altitude_km', 'velocity_ms', 'alpha_deg']].values.astype(np.float32)
Y_raw = df[['CL_mean', 'CD_mean', 'q_mean']].values.astype(np.float32)

X_mean, X_std = X_raw.mean(0), X_raw.std(0)
Y_mean, Y_std = Y_raw.mean(0), Y_raw.std(0)

X = torch.tensor((X_raw - X_mean) / X_std)
Y = torch.tensor((Y_raw - Y_mean) / Y_std)

n_val    = max(1, int(0.15 * len(X)))
train_ds, val_ds = random_split(TensorDataset(X, Y), [len(X) - n_val, n_val])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=64)

model = EvidentialMLP()
opt   = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(300):
    model.train()
    for xb, yb in train_dl:
        mu, v, alpha, beta = model(xb)
        loss = evidential_loss(mu, v, alpha, beta, yb)
        opt.zero_grad(); loss.backward(); opt.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_dl:
                mu, v, alpha, beta = model(xb)
                vals.append(evidential_loss(mu, v, alpha, beta, yb).item())
        print(f"Epoch {epoch+1:3d} | val loss {np.mean(vals):.4f}")

torch.save({
    'model': model.state_dict(),
    'X_mean': X_mean, 'X_std': X_std,
    'Y_mean': Y_mean, 'Y_std': Y_std,
}, 'checkpoint.pt')
print("Saved checkpoint.pt")
