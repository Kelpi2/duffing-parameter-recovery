# Rediscovering Physics from Noise

**Can a learning algorithm recover hidden physical laws from noisy observations?**

---

## Project Overview

This 30-day solo research project investigates whether learning algorithms, built from scratch in Python and NumPy, can rediscover the parameters governing a nonlinear physical system — the **Duffing oscillator** — from noisy time-series observations alone. The project spans simulation, algorithm implementation, and scientific analysis, producing a complete research narrative suitable for university applications in engineering and physics.

## The Physical System: Duffing Oscillator

The Duffing equation models a nonlinear spring-mass system with damping and periodic driving:

```
x'' + γ x' + α x + β x³ = F cos(ω t)
```

where the five recoverable parameters are:

- **γ** — damping coefficient (energy dissipation rate)
- **α** — linear stiffness (reduces to SHM when β = 0)
- **β** — nonlinear (cubic) stiffness (controls anharmonicity)
- **F** — driving force amplitude
- **ω** — driving angular frequency

When β = 0 the system is a standard damped driven harmonic oscillator (A-level SHM + damping). As β increases, the restoring force becomes nonlinear, producing amplitude-dependent frequency shifts and eventually chaotic motion — giving a natural axis along which to test where learning breaks down.

## Why This System?

- **Validation gateway** — the β = 0 limit is analytically solvable, so you can verify every algorithm against exact results before entering nonlinear territory.
- **Tuneable complexity** — a single parameter (β) moves the system from linear → weakly nonlinear → chaotic, giving a clean experimental axis.
- **Five distinct parameters** — enough for a meaningful inference problem, each with different identifiability properties under noise.
- **Real-world relevance** — the Duffing model appears in structural engineering, MEMS design, and nonlinear vibration analysis.
- **A-level connection** — builds directly on SHM, damping, and driven oscillations from your Physics syllabus.

## Models Built from Scratch

| Model | What it recovers | Key maths |
|-------|-----------------|-----------|
| Linear regression | α, γ from linear regime data | Normal equation, gradient descent |
| AR time-series model | Discrete dynamics parameters, short-horizon prediction | Yule–Walker, AIC model selection |
| MLP (manual backprop) | Full nonlinear mapping x(t) → x(t+Δt) including β, F, ω effects | Chain rule, SGD, ReLU/tanh |

## Scientific Questions

- How does observation noise affect parameter recovery accuracy for each model?
- Which parameters are easiest/hardest to recover, and why?
- At what noise level or nonlinearity strength does each model break down?
- Can an MLP learn the nonlinear dynamics that linear methods cannot?
- What is the relationship between prediction horizon and model accuracy?
- Does the system's proximity to chaos create a fundamental inference barrier?

## Prerequisites and Background Assumed

A-level calculus (differentiation, integration, chain rule), basic linear algebra (matrix multiplication, inverses), Python with NumPy, familiarity with gradient descent and loss functions from prior project work. New maths (Runge-Kutta, backpropagation, Yule-Walker) is scheduled into the plan below.

---

## Phase 1 — Foundations & Simulation (Days 1–7)

**Goal:** understand the Duffing oscillator mathematically, build a reliable numerical simulator, generate clean and noisy datasets, and implement linear regression as the first learning algorithm.

### Day 1: Maths Review — ODEs and SHM

- Revise second-order ODEs: convert x'' + γx' + αx = 0 into a first-order system of two equations
  ```
  Let v = x'   →   v' = −γv − αx − βx³ + F cos(ωt)
  ```
- Work through the exact solution for the undamped case (β = 0, γ = 0): x(t) = A cos(ω₀t + φ)
- Derive the damped solution categories: underdamped, critically damped, overdamped
- Sketch phase portraits (x vs v) for each regime by hand

> **Deliverable:** handwritten notes covering the ODE theory. These form the 'ground truth' you'll test your algorithms against.

### Day 2: Numerical Integration — Euler and RK4

- Implement forward Euler: y_{n+1} = y_n + h·f(t_n, y_n) for the 2D system [x, v]
- Implement 4th-order Runge-Kutta (RK4) — derive or follow the standard k1–k4 formula
- Compare Euler vs RK4 on the linear oscillator (β = 0) against the exact analytical solution
- Measure energy drift: compute E = ½v² + ½αx² over 1000 periods with each method
- Choose a timestep (aim for dt ≈ 0.01 of the natural period) that keeps RK4 energy error below 0.1%

> **Deliverable:** simulator.py with rk4_step() and simulate() functions. Plots of Euler vs RK4 accuracy.

### Day 3: Simulator Validation & Parameter Exploration

- Validate the full Duffing simulator (β > 0) by checking conservation laws in limiting cases
- Generate phase portraits for β = 0, 0.1, 0.5, 1.0, 5.0 — observe how orbits distort
- Sweep driving frequency ω near resonance (ω ≈ √α) and plot amplitude response curves
- Identify the parameter regime where the system transitions to chaos (large β, moderate F)
- Create a parameter configuration file: define 'easy', 'medium', and 'hard' parameter sets

> **Deliverable:** validated simulator with parameter configs. Gallery of phase portraits and frequency response plots.

### Day 4: Noise Model & Dataset Generation

- Add Gaussian observation noise: x_obs(t) = x_true(t) + ε, where ε ~ N(0, σ²)
- Implement signal-to-noise ratio (SNR) control: SNR = σ_signal / σ_noise
- Generate datasets at SNR = 100, 10, 5, 2, 1 for the linear case (β = 0)
- Plot noisy vs clean trajectories side by side; compute and store ground-truth parameters with each dataset
- Also generate finite-difference velocity estimates: v_est(t) ≈ (x(t+dt) − x(t−dt)) / 2dt

> **Deliverable:** data_generator.py producing .npz dataset files. Noise level visualisation plots.

### Day 5: Linear Regression — Normal Equation

- Frame parameter recovery as linear regression: from the ODE x'' = −γx' − αx, if you estimate x'' and x' from data, then x'' ≈ [−γ, −α] · [x', x]ᵀ
- Implement the normal equation: θ = (XᵀX)⁻¹ Xᵀy
- Apply to the β = 0 dataset at SNR = 100: recover α and γ, compare to ground truth
- Compute percentage error for each recovered parameter
- Analyse how finite-difference noise amplification affects the velocity and acceleration estimates

> **Deliverable:** linear_regression.py with normal equation solver. First parameter recovery results table.

### Day 6: Linear Regression — Gradient Descent

- Implement batch gradient descent for the same regression problem: MSE loss, manual gradient computation
- Implement learning rate scheduling (start at 0.01, decay by 0.95 every 100 steps)
- Plot the loss curve and parameter convergence trajectories over training iterations
- Compare final parameters and convergence speed against the normal equation solution
- Add L2 regularisation (Ridge): observe its effect on recovered parameters at high noise

> **Deliverable:** gradient_descent.py. Loss curves and parameter convergence plots. Comparison table: normal eq vs GD vs Ridge.

### Day 7: Noise Sensitivity Study (Linear Model)

- Run linear regression (normal equation) across all five SNR levels
- Plot recovered α and γ vs SNR with error bars (repeat each experiment 20 times with different noise seeds)
- Derive the theoretical noise floor: how does finite-difference error scale with σ and dt?
- Identify the SNR threshold below which parameter recovery fails (error > 20%)
- Write up Phase 1 results: clean summary of what linear regression can and cannot do

> **Deliverable:** Phase 1 results notebook. Key plot: parameter error vs SNR. Written summary (½ page).

---

## Phase 2 — Time-Series Modelling / AR (Days 8–13)

**Goal:** build an autoregressive model that learns discrete-time dynamics directly from observed sequences, bypassing the need for noisy derivative estimates. Compare its parameter recovery and predictive accuracy against linear regression.

### Day 8: AR Model Theory

- Study autoregressive models: x(t) = Σ a_k · x(t − k·Δt) + ε(t) for k = 1…p
- Understand the connection between AR coefficients and the underlying continuous ODE
- Derive the Yule-Walker equations: solve for AR coefficients from the autocorrelation matrix
- Work through the mapping: for a second-order ODE discretised at step dt, an AR(2) model's coefficients encode α, γ

> **Deliverable:** handwritten derivation of Yule-Walker equations and the AR ↔ ODE coefficient mapping.

### Day 9: AR Model Implementation

- Implement AR(p) fitting via Yule-Walker: compute autocorrelation, solve the Toeplitz system
- Implement AR(p) fitting via least-squares (for comparison and as a check)
- Test on the linear oscillator (β = 0, SNR = 100): fit AR(2) and extract α, γ from the coefficients
- Verify the recovered parameters match ground truth to within numerical precision

> **Deliverable:** ar_model.py with fit() and predict() methods. Verification on clean linear data.

### Day 10: Model Order Selection

- Implement AIC (Akaike Information Criterion) to select optimal AR order p
- Fit AR(1) through AR(10) on the linear oscillator data; plot AIC vs p — expect minimum at p = 2
- Repeat for the nonlinear case (β > 0): observe that AIC selects higher p to approximate nonlinearity
- Discuss the bias-variance tradeoff: underfitting (p too low) vs overfitting (p too high)

> **Deliverable:** AIC plots for linear and nonlinear cases. Discussion of model order selection.

### Day 11: AR Prediction & Multi-Step Forecasting

- Implement one-step-ahead prediction: given x(t−pΔt)…x(t−Δt), predict x(t)
- Implement multi-step (recursive) forecasting: feed predictions back as inputs
- Plot predicted vs actual trajectories for horizons of 1, 10, 50, 200 steps
- Compute RMSE as a function of prediction horizon — expect exponential error growth in nonlinear regime
- Compare prediction accuracy: linear regime vs nonlinear regime

> **Deliverable:** forecasting plots. RMSE vs horizon curves for linear and nonlinear cases.

### Day 12: AR Noise Sensitivity & Nonlinear Limits

- Run AR(2) parameter recovery across all five SNR levels (linear case, 20 repetitions each)
- Compare AR noise sensitivity against linear regression from Phase 1 on the same data
- Test AR on weakly nonlinear data (small β): how well does a linear AR approximate the dynamics?
- Quantify the 'linearisation error': fit AR to nonlinear data, compare recovered 'effective α' to true α
- Plot the model residuals — they should reveal the missing cubic term as a structured pattern

> **Deliverable:** comparative noise sensitivity plots (AR vs LinReg). Residual analysis showing AR's linear assumption breaking.

### Day 13: Phase 2 Write-Up & Comparison

- Compile all AR results into a clean summary
- Create a comparison table: LinReg vs AR across all metrics (parameter error, prediction RMSE, noise robustness)
- Key insight to articulate: AR avoids derivative estimation noise but is fundamentally linear
- Identify the gap: neither model can capture the nonlinear dynamics — motivating the MLP
- Plan the MLP architecture based on what you now know about the problem's difficulty

> **Deliverable:** Phase 2 summary (½ page). Comparison table. Clear statement of what the MLP needs to do that AR cannot.

---

## Phase 3 — Neural Network / MLP (Days 14–22)

**Goal:** build a multilayer perceptron entirely from scratch (forward pass, backpropagation, SGD) and train it to learn the full nonlinear dynamics of the Duffing oscillator. This is the most technically challenging phase.

### Day 14: MLP Theory — Forward Pass

- Define the architecture: input layer (window of past states) → hidden layers → output (next state)
- Concrete starting architecture: [2p inputs] → [32 neurons] → [16 neurons] → [2 outputs (x, v)]
- Implement activation functions: ReLU, tanh, and their derivatives (you'll need these for backprop)
- Implement the forward pass: z = Wx + b, a = σ(z) for each layer
- Initialise weights using He initialisation (for ReLU) or Xavier (for tanh)

> **Deliverable:** mlp.py with Layer class, forward() method, and weight initialisation.

### Day 15: MLP Theory — Backpropagation

- Derive backpropagation from the chain rule: ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
- Work through the full backward pass on paper for a 2-layer network with MSE loss
- Implement backward() for each layer: compute gradients for weights, biases, and pass δ upstream
- Verify with numerical gradient checking: (L(w+ε) − L(w−ε)) / 2ε ≈ analytical gradient
- Gradient check must pass to < 10⁻⁵ relative error before proceeding

> **Deliverable:** backward() method with gradient checking utility. Handwritten backprop derivation.

### Day 16: Training Loop & SGD

- Implement mini-batch SGD: shuffle data, split into batches of 64, update weights after each batch
- Implement the training loop: epochs, batch iteration, loss tracking
- Add learning rate decay and simple early stopping (stop if validation loss hasn't improved in 20 epochs)
- Train on the linear oscillator (β = 0, SNR = 100) as a sanity check — the MLP should match AR performance
- Plot training and validation loss curves; verify no obvious overfitting

> **Deliverable:** training loop with SGD. Loss curves on linear data showing successful learning.

### Day 17: MLP on Nonlinear Data

- Train the MLP on weakly nonlinear data (small β) — the regime where AR failed
- Compare MLP predictions against AR predictions on the same test set
- Visualise: overlay MLP predicted trajectory, AR predicted trajectory, and ground truth
- If MLP outperforms AR, you've demonstrated the value of nonlinear function approximation
- If MLP struggles, investigate: is the issue architecture, learning rate, or data quantity?

> **Deliverable:** first nonlinear results. MLP vs AR prediction comparison plots.

### Day 18: Hyperparameter Tuning

- Systematically vary: hidden layer sizes [16, 32, 64], number of layers [1, 2, 3], activation [ReLU, tanh]
- For each configuration, train for 500 epochs and record best validation loss
- Create a results table of architecture vs performance
- Select the best architecture for the remaining experiments
- Investigate the input representation: does feeding [x, v] vs [x(t), x(t−dt), x(t−2dt),...] matter?

> **Deliverable:** hyperparameter search results table. Selected architecture with justification.

### Day 19: MLP Noise Robustness

- Train the MLP at each SNR level (100, 10, 5, 2, 1) on both linear and nonlinear data
- Compute prediction RMSE and compare against LinReg and AR at every noise level
- Create a unified 3-model comparison plot: RMSE vs SNR for all three methods
- Key question: does the MLP's extra capacity help or hurt at high noise? (overfitting risk)
- Add dropout (implemented manually: randomly zero out neurons during training) if overfitting is observed

> **Deliverable:** 3-model noise robustness comparison. Dropout implementation if needed.

### Day 20: Parameter Extraction from MLP

- Unlike LinReg and AR, the MLP doesn't directly output physical parameters — they're encoded implicitly
- Approach 1: Jacobian analysis — compute ∂output/∂input numerically; in the linear regime, this Jacobian should encode α and γ
- Approach 2: Symbolic probing — feed carefully designed synthetic inputs (e.g., x only, v only) and measure responses to isolate parameter effects
- Approach 3: Compare MLP-learned dynamics against the ODE by evaluating the MLP as a vector field and overlaying with the true vector field
- Discuss the interpretability gap: the MLP learns to predict accurately but doesn't give you a formula

> **Deliverable:** Jacobian-based parameter extraction results. Vector field comparison plots. Discussion of interpretability.

### Day 21: Chaos Boundary Experiment

- The key experiment: sweep β from 0 to the chaotic regime while keeping other parameters fixed
- At each β value: generate data, train all three models, compute prediction RMSE at horizon = 100 steps
- Plot RMSE vs β for all three models — expect a sharp transition where all models fail
- Compute the Lyapunov exponent (numerically, by tracking divergence of nearby trajectories) to mark the chaos onset
- Correlate: does the prediction breakdown coincide with positive Lyapunov exponent?

> **Deliverable:** the project's headline plot — prediction error vs nonlinearity strength, annotated with chaos boundary.

### Day 22: Phase 3 Write-Up

- Compile all MLP results into a summary
- Create the full three-model comparison across all axes: noise, nonlinearity, prediction horizon
- Articulate the key findings: where each model excels, where it fails, and why
- Draft the 'conclusions so far' section — what have you learned about the limits of learning from noisy observations?

> **Deliverable:** Phase 3 summary (1 page). Complete model comparison figures and tables.

---

## Phase 4 — Analysis, Polish & Write-Up (Days 23–30)

**Goal:** deepen the scientific analysis, produce publication-quality figures, write the final research report, and prepare materials for your personal statement.

### Day 23: Information-Theoretic Analysis

- Study the Cramér-Rao bound: the theoretical minimum variance for any unbiased parameter estimator
- Compute the Fisher information matrix for the linear oscillator model given Gaussian noise at level σ
- Compare your actual estimation errors (from all three models) against the Cramér-Rao lower bound
- This tells you: are your algorithms limited by the data, or by their own capacity?

> **Deliverable:** Fisher information calculations. Plot comparing empirical errors to the theoretical floor.

### Day 24: Prediction Horizon Analysis

- For each model and each nonlinearity level, compute RMSE as a function of prediction horizon (1–500 steps)
- In the chaotic regime, the error should grow exponentially — fit an exponential to measure the effective Lyapunov time
- Compare: does the MLP extend the 'useful prediction horizon' compared to AR?
- Create a heatmap: (model × nonlinearity × horizon) → prediction accuracy

> **Deliverable:** prediction horizon heatmap. Lyapunov time comparison across models.

### Day 25: Which Parameters Are Hardest to Recover?

- Systematic parameter identifiability study: for each of the five Duffing parameters, compute recovery error across all models and noise levels
- Rank parameters from most to least recoverable
- Physical interpretation: why is the driving frequency ω easy to find (it's in the Fourier spectrum) while β might be hard (its effect is subtle until strongly nonlinear)?
- Create a parameter identifiability matrix: (parameter × model × noise level) → recovery success/failure

> **Deliverable:** parameter identifiability analysis. Ranked list with physical explanations.

### Day 26: Publication-Quality Figures

- Redesign all key plots using matplotlib with consistent styling, proper axis labels, legends, and annotations
- Create the '6 figures that tell the story': (1) system overview/phase portraits, (2) parameter recovery vs noise, (3) AR vs LinReg comparison, (4) MLP nonlinear advantage, (5) chaos boundary plot, (6) prediction horizon heatmap
- Export all figures as high-resolution PNGs and vector PDFs
- Ensure every figure has a caption that a reader could understand without the main text

> **Deliverable:** 6 polished figures with captions. Consistent visual style across all plots.

### Day 27: Final Report — Introduction & Methods

- Write the report introduction (1 page): motivation, research question, why this matters
- Write the methods section (2 pages): the Duffing system, simulation approach, all three learning algorithms
- Include the key equations, architecture diagrams, and hyperparameter choices
- Target audience: an admissions tutor with a physics/engineering background

> **Deliverable:** draft of Introduction and Methods sections (~3 pages).

### Day 28: Final Report — Results & Discussion

- Write the results section (2–3 pages): present each experiment with figures and tables
- Write the discussion (1–2 pages): interpret the results, connect to physics, acknowledge limitations
- Key narrative: 'linear methods work until nonlinearity matters; the MLP extends the frontier but chaos sets a hard limit; noise amplification through derivatives is a universal bottleneck'

> **Deliverable:** draft of Results and Discussion sections (~4 pages).

### Day 29: Final Report — Conclusions, Abstract, References

- Write conclusions (½ page): three bullet-point findings + what you'd do next with more time
- Write the abstract (200 words): the entire project in miniature
- Add references: cite the Duffing equation, Runge-Kutta, backpropagation, Cramér-Rao bound
- Proofread the full report end-to-end; check all figures are referenced in the text

> **Deliverable:** complete research report (~8–10 pages plus figures).

### Day 30: Personal Statement Material & Code Cleanup

- Write a 150-word personal statement paragraph summarising this project and what it demonstrates about you
- Clean up the codebase: add docstrings, a README, and a requirements.txt (just numpy + matplotlib)
- Create a GitHub repository with a clear structure: /src, /data, /figures, /report
- Run all experiments one final time from a clean state to ensure reproducibility
- Record a 3-minute screen recording walking through the key results (optional but impressive for interviews)

> **Deliverable:** personal statement paragraph. Clean GitHub repo. Reproducible experiment pipeline.

---

## Appendix: Maths You'll Need (Quick Reference)

### Runge-Kutta 4th Order (Day 2)

Given dy/dt = f(t, y), one step of size h:

```
k1 = f(t, y)
k2 = f(t + h/2,  y + h·k1/2)
k3 = f(t + h/2,  y + h·k2/2)
k4 = f(t + h,  y + h·k3)
y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
```

### Normal Equation (Day 5)

For y = Xθ + ε, the least-squares solution is:

```
θ* = (XᵀX)⁻¹ Xᵀy
```

With L2 regularisation:

```
θ* = (XᵀX + λI)⁻¹ Xᵀy
```

### Yule-Walker Equations (Day 8)

For an AR(p) model, the autocorrelation method gives:

```
R · a = r
```

where R is the p×p autocorrelation matrix and r = [r(1), r(2), ..., r(p)]ᵀ is the autocorrelation vector.

### Backpropagation (Day 15)

For layer l with weights W, input a_{l−1}, pre-activation z = Wa_{l−1} + b:

```
∂L/∂W = δ · a_{l−1}ᵀ
∂L/∂b = δ
δ_{l−1} = (Wᵀ · δ) ⊙ σ'(z_{l−1})    [element-wise multiply with activation derivative]
```

### Fisher Information & Cramér-Rao Bound (Day 23)

```
Fisher information: I(θ) = −E[∂² ln L(θ) / ∂θ²]
Cramér-Rao: Var(θ̂) ≥ 1 / I(θ)  for any unbiased estimator θ̂
```

This gives the theoretical minimum estimation error achievable from noisy data.

---

## Suggested Repository Structure

```
physics-from-noise/
├── src/
│   ├── simulator.py         # RK4 solver, Duffing ODE, noise model
│   ├── data_generator.py    # Dataset creation at various SNR levels
│   ├── linear_regression.py # Normal equation + gradient descent
│   ├── ar_model.py          # AR fitting, Yule-Walker, AIC
│   ├── mlp.py               # MLP with manual backprop
│   ├── experiments.py       # All experiment runners
│   └── plotting.py          # Publication-quality figure generation
├── data/                    # Generated .npz datasets
├── figures/                 # Output plots
├── report/                  # LaTeX or Markdown report
├── notebooks/               # Exploration / scratch work
├── README.md
└── requirements.txt         # numpy, matplotlib
```

---

*This plan is designed for roughly 3–4 hours of focused work per day. Days are flexible — if a task takes longer, shift subsequent days accordingly. The important thing is completing each phase's deliverables before moving to the next. The scientific narrative matters more than hitting every detail.*
