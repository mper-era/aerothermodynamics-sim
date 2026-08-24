import numpy as np
from scipy.stats import qmc
import pandas as pd
from pathlib import Path

Path('data').mkdir(exist_ok=True)

ood_groups = {
    'high_altitude': {
        'bounds': {
            'altitude_km': (125, 145),
            'velocity_ms': (10500, 13000),
            'alpha_deg':   (0, 20),
            'Twall_K':     (300, 1500),
        },
        'seed': 99,
    },
    'low_altitude': {
        'bounds': {
            'altitude_km': (10, 29),
            'velocity_ms': (3000, 10000),
            'alpha_deg':   (0, 20),
            'Twall_K':     (300, 1500),
        },
        'seed': 101,
    },
    'shifted_velocity': {
        'bounds': {
            'altitude_km': (30, 120),
            'velocity_ms': (10000, 12000),
            'alpha_deg':   (15, 20),
            'Twall_K':     (300, 1500),
        },
        'seed': 102,
    },
}

N = 30
all_dfs = []
for name, spec in ood_groups.items():
    bounds = spec['bounds']
    sampler = qmc.LatinHypercube(d=4, seed=spec['seed'])
    samples = qmc.scale(sampler.random(N), [v[0] for v in bounds.values()], [v[1] for v in bounds.values()])
    df = pd.DataFrame(samples, columns=bounds.keys())
    df['ood_source'] = name
    all_dfs.append(df)
    print(f"Generated {N} OOD samples for '{name}'")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv('data/ood_samples.csv', index=False)
print(f"\nTotal: {len(df_all)} OOD samples across {len(ood_groups)} groups")
print(df_all.groupby('ood_source').describe().round(2))