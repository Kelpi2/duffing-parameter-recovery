# Physics from Noise — Project Context

## Status
- **Current phase:** Phase 2 — Time-Series Modelling / AR (Days 8–13)
- **Current day:** Days 9, 11, 12 coded. Day 10 partial (AIC works, nonlinear run outstanding). Next: Phase 3 MLP.
- **Plan change (2026-07-08):** write-up and derivations (noise-floor derivation, Phase 1 write-up) are deferred until after all the coding is done. Focus is on the coding first. Figures are generated and inspected but not saved/styled — that is Day 26 work.
- **Outstanding from Phase 1:** normal eq vs GD vs Ridge comparison table (Day 6).
- **Last updated:** 2026-07-30

## System
Duffing oscillator: `x'' + γx' + αx + βx³ = F cos(ωt)`

Five recoverable parameters: γ (damping), α (linear stiffness), β (nonlinear stiffness), F (driving amplitude), ω (driving frequency)

First-order form (state vector [x, v]):
```
x' = v
v' = −γv − αx − βx³ + F cos(ωt)
```

## Behaviour
Do not make any edits yourself unless specifically requested. Guidance should be provided as default response.

The Session log is written by Maciej in his own words — do not write log entries for him. If something happens that seems worth logging (progress, or something that tripped him up) and he hasn't logged it, just remind him.

## Constraints
- Python + NumPy only (no ML libraries)
- All algorithms built from scratch (linear regression, AR model, MLP with manual backprop)
- matplotlib allowed for plotting only

## Repo structure
```
src/                  → simulator.py, generator.py, linear_regression.py, gradient_decent.py, ar_model.py
                        (planned: mlp.py, experiments.py, plotting.py)
data/                 → generated .npz datasets (gitignored)
figures/              → output plots
report/               → final write-up
docs/30_day_plan.md   → full 30-day plan
```

## Key decisions
- dt = 0.063 (0.01 × T for α=1). RK4 energy error ~0.005% — well under 0.1% target.

## Session log
- Day 1 completed — handwritten ODE notes
- Day 2 completed — simulator built, Euler vs RK4 vs analytical compared. Phase portrait and energy drift plots done.
- Day 3 completed — energy conservation validated, phase portraits for β sweep, omega sweep with resonance peak, chaos confirmed. Easy/medium/hard parameter configs defined.
- Day 4 completed — Gaussian noise model, SNR control, datasets saved as .npz for SNR 100/10/5/2/1, FDV velocity estimation, noisy vs clean phase portrait visualisation.
- Day 5 completed — linear_regression.py built. buildMatrices() runs second FDV for acceleration, normalEq() implements normal equation. Recovers α and γ across all SNR levels. Key finding: α degrades badly with noise due to double FDV amplification; γ stays stable because its predictor (v) shares the same noise source as y (a).
- Day 6 (in progress) — gradient_decent.py built: batch GD with lr decay (×0.95 per 100 epochs), L2 regularisation, feature normalisation, loss/convergence curves. NoiseStudy (Day 7 deliverable) done in linear_regression.py: 20-repeat error bars for α, γ vs SNR. Outstanding: normal eq vs GD vs Ridge comparison table, noise-floor derivation, Phase 1 write-up.

## What's been built
- `src/simulator.py` — duffing ODE, euler_step, RK4, simulate functions, analytical solution, energy calculation. Param sets: linear_params (F=0, γ=0.2), easy_params, medium_params, hard_params.
- `src/generator.py` — addNoise(), FDV(), generateDataset() saves .npz files with NoisyDis, NoisyVel, CleanStates, timestep. Datasets generated using linear_params (F=0, γ=0.2) to avoid driving force corrupting regression.
- `src/linear_regression.py` — buildMatrices(), normalEq(), linearReg() loops over SNR levels and prints recovery table. NoiseStudy() repeats recovery 20× per SNR and plots error bars.
- `src/gradient_decent.py` — loss(), grad(), gradient_descent() with lr decay, L2, and normalised features.
- `src/ar_model.py` — fit() does AR(p) least squares and returns coefficients + AIC. recoverParam() maps AR(2) coefficients back to α and γ via a2 = −e^(−γh), a1 = 2e^(−γh/2)cos(ω_d·h), with a guard for invalid a2/arccos values. AIC() sweeps order 1–Maxp. predict() does recursive multi-step forecasting and builds RMSE vs horizon across sliding start points. All three take a decimation argument.

## Key decisions (additions)
- Datasets generated with F=0 (linear_params) for regression — driving force term not in regression model so must be zero to avoid bias.
- X matrix column order: [noisyDis, noisyVel] → solutions[0]=α, solutions[1]=γ (negated).
- camelCase naming convention throughout all files.
- **AR fits use decimation (dec = 25).** dt = 0.063 was chosen for RK4 accuracy, not identifiability. At that spacing a2 → −1 and the arccos argument → 1 almost regardless of γ and α, so noise swamps what little parameter signal remains — every fit fails below SNR 100 (a2 turns positive, −ln(−a2) undefined). Error minimises around dec 25–30; beyond that sample count drops too far and it rises again. Simulate finely, fit coarsely.
- Datasets are regenerated per parameter set rather than tagged by filename — switching between linear and nonlinear means rerunning `generateDataset`, and only one set exists on disk at a time.
- AIC is run on noisy data, not clean: on clean data the residuals are RK4 rounding error and XᵀX is rank-deficient past p=2.
- Forecasting scores noisy input against clean truth. `NoisyDis[j]` corresponds to `CleanStates[j+1, 0]` because of the `[1:-1]` trim in generator.py.
