# Offline Working Plan — Days 5–7

Use this alongside `docs/30_day_plan.md` for full context.
Your parameter configs are in `simulator.py`: `easy_params`, `medium_params`, `hard_params`.
Your datasets are in `data/` as `dataset_SNR100.npz` etc.
Load them with `np.load("path/to/dataset_SNR100.npz")` — keys are `CleanStates`, `NoisyDis`, `NoisyVel`.

---

## Day 5 — Linear Regression: Normal Equation
**New file:** `src/linear_regression.py`

### What you're doing
Recovering γ and α from noisy data using the ODE structure.

From `x'' = -γx' - αx` (β=0 case), frame as linear regression:
- Feature matrix `X`: two columns — estimated velocity `x'` and displacement `x`
- Target `y`: estimated acceleration `x''`
- Parameters to find: `θ = [-γ, -α]`

### Steps

**1 — Get derivative estimates**

Import `FDV` from `generator.py` and apply it twice:
- `vel_est = FDV(noisy_x, dt)` — estimated velocity
- `acc_est = FDV(vel_est, dt)` — estimated acceleration

Each FDV call removes 2 points from the ends. So:
- `noisy_x` has n points
- `vel_est` has n-2 points
- `acc_est` has n-4 points

Trim everything to match `acc_est` length:
- `x_trim = noisy_x[2:-2]` — trim 2 from each end
- `vel_trim = vel_est[1:-1]` — trim 1 from each end

**2 — Build X and y**

```
X = np.column_stack((vel_trim, x_trim))   # shape (n-4, 2)
y = acc_est                                # shape (n-4,)
```

**3 — Normal equation**

```
θ = (XᵀX)⁻¹ Xᵀy
```

In NumPy:
```python
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

`theta[0]` ≈ -γ, `theta[1]` ≈ -α. Negate to get recovered γ and α.

**4 — Test on SNR=100**

Load `dataset_SNR100.npz`, run the normal equation, compare to `easy_params`:
```
gamma_error = abs(recovered_gamma - true_gamma) / true_gamma * 100
alpha_error = abs(recovered_alpha - true_alpha) / true_alpha * 100
```

Print a results table. At SNR=100 errors should be small (under 5%).

**5 — Run across all SNR levels**

Loop over all 5 datasets, recover parameters each time, record errors. Print or store as a table.

**Key insight to observe:** error grows as SNR drops. Acceleration estimates (double FDV) are particularly noisy — dividing by `(2*dt)²` amplifies noise badly.

---

## Day 6 — Linear Regression: Gradient Descent
**Add to:** `src/linear_regression.py`

### What you're doing
Solving the same regression problem but iteratively instead of directly.

**1 — MSE loss**

```
L = (1/n) * sum((X @ theta - y)²)
```

Gradient:
```
dL/dtheta = (2/n) * Xᵀ(X @ theta - y)
```

**2 — Gradient descent loop**

```python
theta = np.zeros(2)
lr = 0.01
for epoch in range(n_epochs):
    grad = (2/n) * X.T @ (X @ theta - y)
    theta = theta - lr * grad
```

**3 — Learning rate decay**

Multiply lr by 0.95 every 100 epochs:
```python
if epoch % 100 == 0:
    lr *= 0.95
```

**4 — Track and plot**

Store loss at each epoch, plot loss curve. Also store `theta` at each epoch and plot parameter convergence — you should see both converge toward the normal equation solution.

**5 — L2 regularisation (Ridge)**

Add a penalty term to prevent overfitting at high noise:
```
θ* = (XᵀX + λI)⁻¹ Xᵀy
```

In NumPy:
```python
theta_ridge = np.linalg.inv(X.T @ X + lam * np.eye(2)) @ X.T @ y
```

Try λ = 0.001, 0.01, 0.1 and see how recovered parameters change at low SNR.

**Deliverable:** loss curves, parameter convergence plots, comparison table: normal equation vs GD vs Ridge.

---

## Day 7 — Noise Sensitivity Study
**Add to:** `src/linear_regression.py` or a new `src/experiments.py`

### What you're doing
Systematic study of how noise affects parameter recovery.

**1 — Repeat each experiment 20 times**

For each SNR level, generate 20 different noise realisations (different random seeds) and recover parameters each time. Store all 20 recovered γ and α values.

To use different seeds:
```python
np.random.seed(seed)  # call before addNoise each time
```

**2 — Compute mean and standard deviation**

```python
mean_gamma = np.mean(recovered_gammas)
std_gamma  = np.std(recovered_gammas)
```

**3 — Plot with error bars**

```python
plt.errorbar(snr_levels, mean_gammas, yerr=std_gammas)
```

Do this for both γ and α. Add a horizontal line for the true value.

**4 — Find the SNR threshold**

Identify the SNR below which error exceeds 20%. That's your "this method breaks down here" threshold.

**5 — Theoretical noise floor (optional but good)**

Double FDV amplifies noise by factor `1/(2*dt)²`. So:
```
σ_acc ≈ σ_noise / (2*dt)²  × √2   (roughly)
```

Compare this theoretical amplification to what you observe empirically.

**6 — Phase 1 write-up (½ page)**

Write a short summary covering:
- What linear regression can recover and under what conditions
- Where it breaks down and why (derivative noise amplification)
- What SNR threshold you found
- Motivation for the AR model (avoids needing derivatives)

Save it in `report/phase1_summary.md`.

---

## Quick reference — things to import

In `linear_regression.py`:
```python
import sys
sys.path.append("c:/VS-Code Main/duffing-parameter-recovery/src")
from simulator import easy_params
from generator import FDV, addNoise
import numpy as np
import matplotlib.pyplot as plt
```

Loading a dataset:
```python
data = np.load("c:/VS-Code Main/duffing-parameter-recovery/data/dataset_SNR100.npz")
clean  = data["CleanStates"]   # shape (n, 2) — columns are [x, v]
noisy_x = data["NoisyDis"]    # shape (n-2,) — already trimmed by FDV once
noisy_v = data["NoisyVel"]    # shape (n-2,) — FDV of noisy_x
```

Note: `NoisyDis` and `NoisyVel` are already the same length (you trimmed in generator). For Day 5 you'll apply FDV again to `NoisyVel` to get acceleration — this removes 2 more points, so trim `NoisyDis` by 1 from each end to match.

---

## If you get stuck

- **Normal equation giving nonsense:** check X and y shapes match, check you're trimming correctly
- **GD not converging:** learning rate probably too high, try 0.001
- **Recovered params way off even at SNR=100:** double check FDV trimming — misaligned arrays will give garbage results
- **Import errors:** make sure `sys.path.append` points to your `src/` folder
