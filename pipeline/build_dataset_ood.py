import pandas as pd
import numpy as np
from pathlib import Path

raw = pd.read_csv('data/ood_results.csv').dropna()
print(f'{len(raw)} OOD rows loaded')

df = raw.copy()

# Regime labels
df['Kn'] = df['Kn_fostrad']
df['regime'] = 'continuum'
df.loc[df['Kn'] > 0.01, 'regime'] = 'transitional'
df.loc[df['Kn'] > 1.0,  'regime'] = 'free_molecular'

# Mean across models
for qty in ['CL', 'CD', 'q']:
    cols = [f'{qty}_sc', f'{qty}_krd', f'{qty}_fr', f'{qty}_vd']
    df[f'{qty}_mean'] = df[cols].mean(axis=1)
    df[f'{qty}_std']  = df[cols].std(axis=1)
    df[f'{qty}_cv']   = df[f'{qty}_std'] / (df[f'{qty}_mean'].abs() + 1e-8)

df.to_parquet('data/ood_processed.parquet', index=False)
print(f'Saved to data/ood_processed.parquet')
print(df['regime'].value_counts())
