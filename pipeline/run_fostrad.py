import matlab.engine
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
FOSTRAD = str(REPO_ROOT / 'fostrad')
STL = str((REPO_ROOT / 'fostrad' / 'IXV.00001.stl').resolve())
LHS_PATH = REPO_ROOT / 'data' / 'lhs_samples.csv'
OUT_PATH = REPO_ROOT / 'data' / 'sweep_results.csv'

MODELS = ['sc', 'krd', 'fr', 'vd']
REQUIRED_COLS = ['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']
MODEL_COLS = [f'{q}_{m}' for q in ('CL', 'CD', 'q') for m in MODELS]

df = pd.read_csv(LHS_PATH)
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise SystemExit(
        f"Missing columns in {LHS_PATH}: {missing}\n"
        "Run: python pipeline/generate_lhs.py"
    )

print("Starting MATLAB engine (15-20s)...")
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath(FOSTRAD), nargout=0)
eng.cd(FOSTRAD, nargout=0)

results = []
failures = 0
for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="FOSTRAD sweep")):
    record = row.to_dict()
    kn_val = np.nan

    for m in MODELS:
        try:
            CL, CD, q, Kn = eng.RUN_Function_sweep(
                STL,
                float(row.altitude_km),
                float(row.velocity_ms),
                float(row.alpha_deg),
                m,
                float(row.Twall_K),
                nargout=4,
            )
            record[f'CL_{m}'] = float(CL)
            record[f'CD_{m}'] = float(CD)
            record[f'q_{m}'] = float(q)
            if np.isnan(kn_val):
                kn_val = float(Kn)
        except Exception as e:
            failures += 1
            print(f"\n  Row {idx} model {m} failed: {e}")
            record[f'CL_{m}'] = np.nan
            record[f'CD_{m}'] = np.nan
            record[f'q_{m}'] = np.nan

    record['Kn_fostrad'] = kn_val
    results.append(record)

    if (idx + 1) % 25 == 0:
        pd.DataFrame(results).to_csv(OUT_PATH, index=False)

eng.quit()
out = pd.DataFrame(results)
out.to_csv(OUT_PATH, index=False)

n_nan_rows = out[MODEL_COLS].isna().any(axis=1).sum()
print(f"\nDone. Results in {OUT_PATH}")
print(f"  Rows: {len(out)}")
print(f"  Model call failures: {failures}")
if n_nan_rows:
    print(f"  WARNING: {n_nan_rows} rows have incomplete model outputs")
