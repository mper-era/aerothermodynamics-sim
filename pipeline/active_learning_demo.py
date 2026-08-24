import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))
import numpy as np
import pandas as pd
import torch
import matlab.engine
from scipy.stats import qmc
from evidential_mlp import EvidentialMLP

ROOT = Path(__file__).parent.parent
STL = str(ROOT / 'fostrad' / 'IXV.00001.stl')

N_CANDIDATES = 60   # pool to score, not to run through FOSTRAD
N_PICK = 15         # actually evaluated per strategy

bounds = {
    'altitude_km': (30, 120),
    'velocity_ms': (3000, 10000),
    'alpha_deg':   (0, 20),
    'Twall_K':     (300, 1500),
}

np.random.seed(7)
sampler = qmc.LatinHypercube(d=4, seed=7)
candidates = qmc.scale(sampler.random(N_CANDIDATES),
                        [v[0] for v in bounds.values()], [v[1] for v in bounds.values()])
cand_df = pd.DataFrame(candidates, columns=bounds.keys())

# --- Score candidates by current MLP's epistemic uncertainty ---
ckpt = torch.load(ROOT / 'model' / 'checkpoint.pt', map_location='cpu', weights_only=False)
mlp = EvidentialMLP(in_dim=4)
mlp.load_state_dict(ckpt['model'])
mlp.eval()

X = torch.tensor(((cand_df.values - ckpt['X_mean']) / ckpt['X_std']).astype(np.float32))
with torch.no_grad():
    mu, v, alpha, beta = mlp(X)
ep_std = (beta / (v * (alpha - 1).clamp(min=1e-4))).sqrt().numpy()[:, 2]  # q channel

cand_df['ep_std_q'] = ep_std
uncertainty_picks = cand_df.nlargest(N_PICK, 'ep_std_q').drop(columns='ep_std_q')

np.random.seed(8)
random_picks = cand_df.sample(N_PICK, random_state=8).drop(columns='ep_std_q')

# --- Run FOSTRAD on both sets ---
print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath(str(ROOT / 'fostrad')), nargout=0)
eng.addpath(str(ROOT / 'pipeline'), nargout=0)
eng.cd(str(ROOT / 'fostrad'), nargout=0)

def run_fostrad_batch(df, label):
    rows = []
    for i, row in df.iterrows():
        CD, CL, q, Kn = eng.RUN_Function_sweep(
            STL, float(row['altitude_km']), float(row['velocity_ms']),
            float(row['alpha_deg']), 'krd', float(row['Twall_K']), nargout=4
        )
        rows.append({**row.to_dict(), 'CD': CD, 'CL': CL, 'q': q, 'Kn': Kn})
        print(f"  {label} {len(rows)}/{N_PICK}")
    return pd.DataFrame(rows)

uncertainty_results = run_fostrad_batch(uncertainty_picks, "uncertainty-guided")
random_results = run_fostrad_batch(random_picks, "random")
eng.quit()

uncertainty_results.to_csv(ROOT / 'data' / 'active_learning_uncertainty.csv', index=False)
random_results.to_csv(ROOT / 'data' / 'active_learning_random.csv', index=False)
print("Saved both result sets.")