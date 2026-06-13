import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial import KDTree
from pathlib import Path

ROOT  = Path(__file__).parent.parent
DATA  = ROOT / 'data'

R    = 6.371e6
rho0 = 1.225
H    = 8500
g0   = 9.80665

def atmosphere(h_m):
    return rho0 * np.exp(-h_m / H)

def make_lookup(df):
    pts = df[['altitude_km', 'velocity_ms', 'alpha_deg', 'Twall_K']].values.astype(float)
    std = pts.std(axis=0)
    std[std == 0] = 1.0
    tree = KDTree(pts / std)
    return tree, pts, std, df

def lookup_aero(tree, pts, std, df, h_km, v_ms, alpha, Twall):
    query = np.array([h_km, v_ms, alpha, Twall]) / std
    _, idx = tree.query(query)
    row = df.iloc[idx]
    return float(row['CD_mean']), float(row['CL_mean']), float(row['q_mean'])

def reentry_3dof(t, state, tree, pts, std, df, alpha, Twall, m, S_ref):
    r, v, gamma = state
    h_m  = r - R
    h_km = h_m / 1e3
    if h_km < 30 or v < 100:
        return [0, 0, 0]

    rho   = atmosphere(h_m)
    g     = g0 * (R / r) ** 2
    q_dyn = 0.5 * rho * v**2

    CD, CL, _ = lookup_aero(tree, pts, std, df, h_km, v, alpha, Twall)

    D      = q_dyn * S_ref * CD / m
    L      = q_dyn * S_ref * CL / m
    dr     = v * np.sin(gamma)
    dv     = -D - g * np.sin(gamma)
    dgamma = (L - (g - v**2 / r) * np.cos(gamma)) / (v + 1e-6)
    return [dr, dv, dgamma]

def simulate(df, v_entry, fpa_deg, alpha, Twall, m=300, S_ref=0.5):
    tree, pts, std, _ = make_lookup(df)
    r0     = R + 120e3
    gamma0 = np.radians(fpa_deg)
    state0 = [r0, v_entry, gamma0]

    def hit_ground(t, y, *a):
        return y[0] - R - 30e3
    hit_ground.terminal  = True
    hit_ground.direction = -1

    def climbing(t, y, *a):
        return y[2]   # fpa — stop when it goes positive
    climbing.terminal  = True
    climbing.direction = 1

    sol = solve_ivp(
        reentry_3dof, [0, 1200], state0,
        args=(tree, pts, std, df, alpha, Twall, m, S_ref),
        events=[hit_ground, climbing], max_step=5.0, rtol=1e-4
    )

    h_km = (sol.y[0] - R) / 1e3
    v_ms = sol.y[1]

    records = []
    for i in range(len(sol.t)):
        CD, CL, q = lookup_aero(tree, pts, std, df,
                                 h_km[i], v_ms[i], alpha, Twall)
        records.append({
            'time_s':      sol.t[i],
            'altitude_km': h_km[i],
            'velocity_ms': v_ms[i],
            'alpha_deg':   alpha,
            'Twall_K':     Twall,
            'fpa_deg':     np.degrees(sol.y[2][i]),
            'CD':          CD,
            'CL':          CL,
            'q':           q,
        })
    return pd.DataFrame(records)

if __name__ == '__main__':
    df = pd.read_parquet(DATA / 'processed.parquet')

    np.random.seed(0)
    N           = 300
    v_entries   = np.random.uniform(6000, 10000, N)
    fpa_entries = np.random.uniform(-2, -15, N)
    alphas      = np.random.uniform(0, 20, N)
    Twalls      = np.random.uniform(300, 1500, N)

    all_trajs = []
    skipped   = 0
    for i in range(N):
        traj = simulate(df, v_entries[i], fpa_entries[i], alphas[i], Twalls[i])
        if len(traj) < 5:
            skipped += 1
            continue
        traj['traj_id'] = i
        all_trajs.append(traj)
        if (i + 1) % 50 == 0:
            print(f'Trajectory {i+1}/{N} | steps: {len(traj)} | skipped so far: {skipped}')

    out = pd.concat(all_trajs, ignore_index=True)
    out.to_parquet(DATA / 'trajectories.parquet', index=False)
    print(f'\nSaved {len(out)} timesteps across {len(all_trajs)} trajectories')
    print(f'Skipped {skipped} trajectories (too short)')
    print(out[['altitude_km','velocity_ms','q']].describe().round(2))
