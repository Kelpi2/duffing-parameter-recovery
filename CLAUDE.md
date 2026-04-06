# Physics from Noise — Project Context

## Status
- **Current phase:** Phase 1 — Foundations & Simulation (Days 1–7)
- **Current day:** Day 4
- **Last updated:** 2026-04-06

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
