import matlab.engine
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

FOSTRAD = str(Path(__file__).parent.parent / 'fostrad')
STL     = str(Path(FOSTRAD) / 'IXV.00001.stl')
MODELS  = ['krd', 'vd']
OUT     = 'data/sweep_results.csv'

df = pd.read_csv('data/lhs_samples.csv')

print("Starting MATLAB engine (15-20s)...")
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath(FOSTRAD), nargout=0)
eng.cd(FOSTRAD, nargout=0)

results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="FOSTRAD sweep"):
    record = row.to_dict()
    for m in MODELS:
        try:
            CL, CD, q = eng.RUN_Function_sweep(
                STL,
                float(row.altitude_km),
                float(row.velocity_ms),
                float(row.alpha_deg),
                m,
                nargout=3
            )
            record[f'CL_{m}'] = float(CL)
            record[f'CD_{m}'] = float(CD)
            record[f'q_{m}']  = float(q)
        except Exception as e:
            print(f"\n  Run {i} model {m} failed: {e}")
            record[f'CL_{m}'] = np.nan
            record[f'CD_{m}'] = np.nan
            record[f'q_{m}']  = np.nan

    results.append(record)

    if (i + 1) % 25 == 0:
        pd.DataFrame(results).to_csv(OUT, index=False)

eng.quit()
pd.DataFrame(results).to_csv(OUT, index=False)
print(f"\nDone. Results in {OUT}")
