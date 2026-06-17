# Confusions to Revisit

Things that came up during the project that weren't fully understood at the time.

---

## 1. Why does fitting to noisy targets still recover accurate parameters?

**Context:** Day 6 gradient descent. Loss at SNR 100 was ~0.113 despite parameter recovery being ~1% accurate. The question was: if we're fitting toward noisy acceleration estimates, why do we still land near the true α and γ?

**Key concepts to look into:**
- Zero-mean noise and how it averages out over large datasets
- The difference between bias and variance in estimators
- Errors-in-variables: what happens when noise is in X (predictors) vs y (targets)
- Why γ survives noise better than α (shared noise source between predictor and target)

---
