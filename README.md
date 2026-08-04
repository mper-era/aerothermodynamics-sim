## Hypersonic Reentry World Model with Calibrated Uncertainty

A probabilistic surrogate framework for hypersonic reentry aerothermodynamics combining physics-based simulation (FOSTRAD) with calibrated uncertainty quantification via evidential deep learning and recurrent world models.

This project trains two complementary neural surrogates on FOSTRAD simulation data:

- **Evidential MLP**: a fast, well-calibrated pointwise surrogate for thermal protection system (TPS) design loops, using a Normal-Inverse-Gamma (NIG) uncertainty head for closed-form epistemic variance
- **Simplified RSSM**: a recurrent world model for temporally coherent uncertainty propagation along full reentry trajectories, enabling guidance and Monte Carlo simulation

## Key Results

| Result | Value |
|--------|-------|
| MLP calibration | Near-perfect (reliability diagram tracks diagonal) |
| RSSM heat flux RMSE | 0.59 MW/m² (vs MLP 1.85 MW/m²) |
| MLP C_L RMSE | 0.06 (vs RSSM 0.06) |
| MLP C_D RMSE | 0.17 (vs RSSM 0.32) |
| OOD epistemic ratio (evidential head) | 0.52× (failure mode confirmed) |
| OOD distance ratio (k-NN) | 2.90× (clean separation) |
| Peak uncertainty altitude (RSSM) | 35–50 km (peak heating corridor) |

*replace with table of result-wise comparisons between evidential MLP and simplified RSSM*

## Requirements

- Python 3.11 (conda environment recommended)
- MATLAB with FOSTRAD (for data generation only; not needed for model training/inference)

*add link to FOSTRAD project and relevant citations*

## Installation

```bash
git clone https://github.com/your-repo/reentry-world-model
cd reentry-world-model
conda create -n reentry python=3.11
conda activate reentry
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn pyarrow
```

*replace package list with a `requirements.txt` file*


## Result Reproduction

### Step 1: Generate the dataset

Requires MATLAB with FOSTRAD installed and accessible via `matlab.engine`.

```bash
conda activate reentry
python pipeline/generate_lhs.py # generates data/lhs_samples.csv
python pipeline/run_fostrad.py
python pipeline/build_dataset.py # generates data/processed.parquet
python pipeline/generate_trajectories.py # generates data/trajectories.parquet
```

### Step 2: Generate OOD data

```bash
python pipeline/generate_ood.py # generates data/ood_samples.csv
python pipeline/run_fostrad_ood.py
python pipeline/build_dataset_ood.py # generates data/ood_processed.parquet
```

### Step 3: Train the models

```bash
python model/train.py # trains evidential MLP, saves checkpoint.pt
python model/train_rssm.py # trains simplified RSSM, saves rssm_checkpoint.pt
```

Both use Adam with lr=3e-4 for 300 epochs.

### Step 4: Run the analysis notebook

```bash
conda activate reentry
jupyter notebook notebooks/results.ipynb
```

Run all cells top-to-bottom. Cell 7 (OOD detection) requires `data/ood_processed.parquet` from Step 2.

## Notebook Structure

| Cell | Description |
|------|-------------|
| Cell 1 | Initialization - load data, models, checkpoints |
| Cell 2 | MLP epistemic uncertainty vs altitude |
| Cell 2b | MLP epistemic uncertainty vs Knudsen number |
| Cell 3 | RSSM rollout: single trajectory + inter-trajectory spread |
| Cell 4 | Reliability diagram - MLP vs RSSM calibration |
| Cell 5 | RMSE comparison on held-out trajectories |
| Cell 6 | LHS parameter space coverage pairplot |
| Cell 7 | OOD detection: evidential uncertainty + k-NN distance |
| Cell 8 | RSSM multi-step rollout degradation |

*update notebook structure description when notebook is updated*

## Model Architecture

### Evidential MLP
- 4-layer MLP, hidden dim 64, ReLU activations
- NIG output head: outputs (μ, v, α, β) per output dimension
- Epistemic variance: β / (v(α−1))
- Loss: NIG NLL + λ|y−μ|(2v+α), λ=0.2

### Simplified RSSM
- GRU deterministic path: h_t = GRU(h_{t-1}, o_{t-1})
- Gaussian prior p(z_t|h_t) and posterior q(z_t|h_t, o_t)
- NIG decoder on concatenation (z_t, h_t)
- Loss: L_NIG + β_KL · KL[q||p], β_KL=0.1
- Inference: prior-only rollout (no observations)

## Known Limitations

- **KL annealing**: fixed β_KL=0.1 produces non-monotonic uncertainty growth with rollout horizon; KL annealing is recommended for future work
- **OOD detection**: the NIG evidential head underestimates uncertainty on OOD inputs (ratio 0.52×); k-NN distance is more reliable (ratio 2.90×)
- **Free-molecular sparsity**: only 14/200 training points fall in the free-molecular regime, limiting evaluation there
- **Single vehicle geometry**: all results are for one STL mesh; generalization across vehicle shapes is untested
- **Short trajectories**: mean trajectory length of 23 steps limits the RSSM's ability to learn long-horizon uncertainty accumulation

---

## License

MIT
