import numpy as np
from scipy.stats import qmc
import pandas as pd
from pathlib import Path

Path('data').mkdir(exist_ok=True)

bounds = {
    'altitude_km': (125, 145),
    'velocity_ms': (10500, 13000),
    'alpha_deg':   (0, 20),
    'Twall_K':     (300, 1500),
}

N = 30
sampler = qmc.LatinHypercube(d=4, seed=99)
samples = qmc.scale(sampler.random(N),
                    [v[0] for v in bounds.values()],
                    [v[1] for v in bounds.values()])

df = pd.DataFrame(samples, columns=bounds.keys())
df.to_csv('data/ood_samples.csv', index=False)
print(f"Generated {N} OOD samples")
print(df.describe().round(2))
