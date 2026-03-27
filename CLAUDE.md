# Physics from Noise — Project Context

## Status
- **Current phase:** Phase 1 — Foundations & Simulation (Days 1–7)
- **Current day:** Day 1
- **Last updated:** Not yet started

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
Nothing yet — Day 1 is theory/handwritten notes.

## Today's tasks (Day 1: Maths Review — ODEs and SHM)
- [ ] Revise second-order ODEs: convert x'' + γx' + αx = 0 into first-order system
- [ ] Work through exact solution for undamped case (β = 0, γ = 0): x(t) = A cos(ω₀t + φ)
- [ ] Derive damped solution categories: underdamped, critically damped, overdamped
- [ ] Sketch phase portraits (x vs v) for each regime by hand
- **Deliverable:** handwritten notes covering the ODE theory

## Key decisions
- (none yet)

## Session log
- Day 1 completed
