import matlab.engine
from pathlib import Path

FOSTRAD = str(Path('fostrad').resolve())
STL     = str(Path('fostrad/IXV.00001.stl').resolve())

print("Starting MATLAB engine (15-20s)...")
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath(FOSTRAD), nargout=0)
eng.cd(FOSTRAD, nargout=0)

print("Running single FOSTRAD call...")
print(f"  STL:      {STL}")
print(f"  altitude: 70 km")
print(f"  velocity: 6500 m/s")
print(f"  AoA:      -10 deg")
print(f"  model:    krd")

CL, CD, q = eng.RUN_Function_sweep(STL, 70.0, 6500.0, -10.0, 'krd', nargout=3)  # checks on
print(f"\nResults:")
print(f"  CL     = {CL:.6f}")
print(f"  CD     = {CD:.6f}")
print(f"  q_stag = {q:.2f} W/m^2")

eng.quit()
print("\nSmoke test passed.")
