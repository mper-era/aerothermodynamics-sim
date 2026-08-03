import numpy as np
from scipy.stats import qmc
import pandas as pd
from pathlib import Path

Path('data').mkdir(exist_ok=True)

bounds = {
    'altitude_km': (30, 120),
    'velocity_ms': (3000, 10000),
    'alpha_deg':   (0, 20),
    'Twall_K':     (300, 1500),
}

N = 300 
sampler = qmc.LatinHypercube(d=4, seed=42)
samples = qmc.scale(sampler.random(N), [v[0] for v in bounds.values()], [v[1] for v in bounds.values()])
df = pd.DataFrame(samples, columns=bounds.keys())

n_topup = 40
sampler_fm = qmc.LatinHypercube(d=4, seed=7)
fm_bounds = {
    'altitude_km': (100, 120),
    'velocity_ms': (3000, 10000),
    'alpha_deg':   (0, 20),
    'Twall_K':     (300, 1500),
}

fm_samples = qmc.scale(sampler_fm.random(n_topup), [v[0] for v in fm_bounds.values()], [v[1] for v in fm_bounds.values()])
df_fm = pd.DataFrame(fm_samples, columns=fm_bounds.keys())

df_full = pd.concat([df, df_fm], ignore_index=True)
df_full.to_csv('data/lhs_samples.csv', index=False)
print(f"Generated {len(df_full)} samples ({N} main LHS + {n_topup} free-molecular top-up)")
print(df_full.describe().round(2))