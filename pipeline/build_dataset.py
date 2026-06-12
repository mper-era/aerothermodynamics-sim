import pandas as pd
import numpy as np

df = pd.read_csv('data/sweep_results.csv').dropna()

# Knudsen number proxy — exponential atmosphere mean free path
df['lambda_m'] = 2.37e-5 * np.exp(df['altitude_km'] / 8.5)
df['Kn']       = df['lambda_m'] / 1.0   # L_ref = 1m (IXV approx)

# Flow regime label
df['regime'] = 'continuum'
df.loc[df['Kn'] > 0.1, 'regime'] = 'transitional'
df.loc[df['Kn'] > 10,  'regime'] = 'free_molecular'

# Mean prediction and inter-model disagreement (epistemic uncertainty proxy)
for qty in ['CL', 'CD', 'q']:
    cols = [f'{qty}_krd', f'{qty}_vd']
    df[f'{qty}_mean'] = df[cols].mean(axis=1)
    df[f'{qty}_std']  = df[cols].std(axis=1)
    df[f'{qty}_cv']   = df[f'{qty}_std'] / (df[f'{qty}_mean'].abs() + 1e-8)

df.to_parquet('data/processed.parquet', index=False)
print(f"Saved {len(df)} rows to data/processed.parquet")
print("\nRegime counts:")
print(df['regime'].value_counts())
print("\nMean q_cv by regime:")
print(df.groupby('regime')['q_cv'].mean().round(4))
