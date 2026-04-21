# Physics from Noise — Project Context

## Status
- **Current phase:** Phase 1 — Foundations & Simulation (Days 1–7)
- **Current day:** Day 5 complete, Day 6 next
- **Last updated:** 2026-04-16

## System
Duffing oscillator: `x'' + γx' + αx + βx³ = F cos(ωt)`

Five recoverable parameters: γ (damping), α (linear stiffness), β (nonlinear stiffness), F (driving amplitude), ω (driving frequency)

First-order form (state vector [x, v]):
```
x' = v
v' = −γv − αx − βx³ + F cos(ωt)
```

## Constraints
- Python + NumPy only (no ML libraries)
- All algorithms built from scratch (linear regression, AR model, MLP with manual backprop)
- matplotlib allowed for plotting only

## Repo structure
```
src/           → simulator.py, data_generator.py, linear_regression.py, ar_model.py, mlp.py, experiments.py, plotting.py
data/          → generated .npz datasets (gitignored)
figures/       → output plots
report/        → final write-up
docs/plan.md   → full 30-day plan
```

## What's been built
- `src/simulator.py` — duffing ODE, euler_step, RK4, simulate functions, analytical solution, energy calculation

## Key decisions
- dt = 0.063 (0.01 × T for α=1). RK4 energy error ~0.005% — well under 0.1% target.

## Session log
- Day 1 completed — handwritten ODE notes
- Day 2 completed — simulator built, Euler vs RK4 vs analytical compared. Phase portrait and energy drift plots done.
- Day 3 completed — energy conservation validated, phase portraits for β sweep, omega sweep with resonance peak, chaos confirmed. Easy/medium/hard parameter configs defined.
- Day 4 completed — Gaussian noise model, SNR control, datasets saved as .npz for SNR 100/10/5/2/1, FDV velocity estimation, noisy vs clean phase portrait visualisation.
- Day 5 completed — linear_regression.py built. buildMatrices() runs second FDV for acceleration, normalEq() implements normal equation. Recovers α and γ across all SNR levels. Key finding: α degrades badly with noise due to double FDV amplification; γ stays stable because its predictor (v) shares the same noise source as y (a).

## What's been built
- `src/simulator.py` — duffing ODE, euler_step, RK4, simulate functions, analytical solution, energy calculation. Param sets: linear_params (F=0,γ=0), easy_params, medium_params, hard_params.
- `src/generator.py` — addNoise(), FDV(), generateDataset() saves .npz files with NoisyDis, NoisyVel, CleanStates, timestep. Datasets generated using linear_params (F=0, γ=0.2) to avoid driving force corrupting regression.
- `src/linear_regression.py` — buildMatrices(), normalEq(), linearReg() loops over SNR levels and prints recovery table.

## Key decisions (additions)
- Datasets generated with F=0 (linear_params) for regression — driving force term not in regression model so must be zero to avoid bias.
- X matrix column order: [noisyDis, noisyVel] → solutions[0]=α, solutions[1]=γ (negated).
- camelCase naming convention throughout all files.
