import sys
import time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'model'))
from evidential_mlp import EvidentialMLP
from rssm import SimplifiedRSSM
import matlab.engine

N_WARMUP = 20
N_TRIALS = 500

def benchmark_mlp():
    ckpt = torch.load(ROOT / 'model' / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model = EvidentialMLP()
    model.load_state_dict(ckpt['model'])
    model.eval()

    x = torch.randn(1, 4)  # single query, matches real-time guidance use case

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)

        times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"MLP single-query inference (n={N_TRIALS}):")
    print(f"  mean={times.mean():.4f}ms  std={times.std():.4f}ms  "
          f"p50={np.percentile(times,50):.4f}ms  p99={np.percentile(times,99):.4f}ms")
    return times

def benchmark_rssm_step():
    ckpt = torch.load(ROOT / 'model' / 'rssm_checkpoint.pt', map_location='cpu', weights_only=False)
    model = SimplifiedRSSM()
    model.load_state_dict(ckpt['model'])
    model.eval()

    h = torch.zeros(1, model.hidden_dim)
    o = torch.randn(1, 4)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            mu, sigma = model.prior(h)
            z = mu + torch.randn_like(mu) * sigma
            model.decode(z, h)
            model.gru(o, h)

        times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            mu, sigma = model.prior(h)
            z = mu + torch.randn_like(mu) * sigma
            model.decode(z, h)
            h_new = model.gru(o, h)
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000
    print(f"RSSM single-timestep inference (n={N_TRIALS}):")
    print(f"  mean={times.mean():.4f}ms  std={times.std():.4f}ms  "
          f"p50={np.percentile(times,50):.4f}ms  p99={np.percentile(times,99):.4f}ms")
    return times

def benchmark_fostrad(n_trials=12):
    print("Starting MATLAB engine (one-time cost, excluded from timing)...")
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(str(ROOT / 'fostrad')), nargout=0)
    eng.addpath(str(ROOT / 'pipeline'), nargout=0)
    eng.cd(str(ROOT / 'fostrad'), nargout=0)

    stl = str(ROOT / 'fostrad' / 'IXV.00001.stl')
    alt, vel, aoa, Tw = 70, 6500, -10, 300  # fixed representative point

    # one untimed warmup call (JIT/first-call overhead), doesn't reduce
    # mesh cost since STRATH_A_mb recomputes the mesh every call regardless
    eng.RUN_Function_sweep(stl, float(alt), float(vel), float(aoa), 'krd', float(Tw), nargout=4)

    times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        eng.RUN_Function_sweep(stl, float(alt), float(vel), float(aoa), 'krd', float(Tw), nargout=4)
        times.append(time.perf_counter() - t0)
        print(f"  trial {i+1}/{n_trials}: {times[-1]:.3f}s")

    eng.quit()

    times = np.array(times)  # seconds, not ms — these are orders of magnitude slower
    print(f"\nFOSTRAD single-model-call inference (n={n_trials}):")
    print(f"  mean={times.mean():.3f}s  std={times.std():.3f}s  "
          f"p50={np.percentile(times,50):.3f}s  p99={np.percentile(times,99):.3f}s")
    return times

if __name__ == '__main__':
    benchmark_mlp()
    print()
    benchmark_rssm_step()
    print()
    benchmark_fostrad()