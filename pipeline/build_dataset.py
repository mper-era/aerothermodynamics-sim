import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = REPO_ROOT / 'data' / 'sweep_results.csv'
OUT_PATH = REPO_ROOT / 'data' / 'processed.parquet'

MODELS = ['sc', 'krd', 'fr', 'vd']
MODEL_COLS = [f'{q}_{m}' for q in ('CL', 'CD', 'q') for m in MODELS]

# FOSTRAD regime threshold (STRATH_A_mb.m)
LIM_KN_CONT = 1e-4
LIM_KN_FMF = 100

df = pd.read_csv(SWEEP_PATH)
n_before = len(df)
df = df.dropna(subset=MODEL_COLS)
n_dropped = n_before - len(df)
if n_dropped:
    print(f"Warning: dropped {n_dropped} rows with incomplete model outputs")

if 'Kn_fostrad' in df.columns and df['Kn_fostrad'].notna().all():
    df['Kn'] = df['Kn_fostrad']
    df['Kn_source'] = 'fostrad'
else:
    df['lambda_m'] = 2.37e-5 * np.exp(df['altitude_km'] / 8.5)
    df['Kn'] = df['lambda_m'] / 1.0
    df['Kn_source'] = 'proxy'
    print("Warning: Kn_fostrad missing; using exponential-atmosphere proxy for Kn")

df['regime'] = 'transitional'
df.loc[df['Kn'] <= LIM_KN_CONT, 'regime'] = 'continuum'
df.loc[df['Kn'] >= LIM_KN_FMF, 'regime'] = 'free_molecular'

for qty in ['CL', 'CD', 'q']:
    cols = [f'{qty}_{m}' for m in MODELS]
    df[f'{qty}_mean'] = df[cols].mean(axis=1)
    df[f'{qty}_std'] = df[cols].std(axis=1)
    df[f'{qty}_cv'] = df[f'{qty}_std'] / (df[f'{qty}_mean'].abs() + 1e-8)

df.to_parquet(OUT_PATH, index=False)
print(f"Saved {len(df)} rows to {OUT_PATH}")
print(f"Kn source: {df['Kn_source'].iloc[0]}")
print("\nRegime counts:")
print(df['regime'].value_counts())
print("\nMean q_cv by regime:")
print(df.groupby('regime')['q_cv'].mean().round(4))
