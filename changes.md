# changes.md — pivoting the model

## The Original Plan

Build a from-scratch HMC sampler that infers a posterior over engine degradation parameters and turns that posterior into a full probability distribution over Remaining Useful Life (RUL). NumPy only — no PyMC, Stan, or autograd.

Simplest model that could possibly work:

$$h(t) = \alpha + \beta t \quad \text{(linear health curve)}$$

$$y_t = h(t) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2) \quad \text{(noisy sensor reading)}$$

Priors Gaussian on $(\alpha, \beta)$. Failure declared when $h(t) = \tau$ (the failure threshold). RUL follows by inverting:

$$t_{\text{failure}} = \frac{\tau - \alpha}{\beta}$$

$$\text{RUL} = (t_{\text{failure}} - t_{\text{last}}) \times t_{\text{max}}$$

**Sensor:** s11 (highest correlation with cycle across the training fleet). 

**Normalisation:**
- $s_{11}$ normalised to $[0, 1]$ using training global min/max
- Cycles normalised by $t_{\text{max}} = 362$ (global max cycle across training fleet)

**Priors derived from OLS on all 100 training engines in normalised coordinates:**

$$\mu_\alpha \approx 0.21, \quad \sigma_\alpha \approx 0.15$$

$$\mu_\beta \approx 0.73, \quad \sigma_\beta \approx 0.24$$

$$\sigma_{\text{noise}} = 0.15 \quad \text{(fixed, not inferred)}$$

**Failure threshold:** mean $s_{11}^{\text{norm}}$ at the last cycle across all 100 training engines, $\tau \approx 0.79$. Not 1.0 because most engines die before $s_{11}$ hits the global max.

**HMC tuned on training engine 10:**

$$\varepsilon = 0.015, \quad L = 20, \quad N = 10{,}000, \quad N_{\text{burn}} = 1{,}000$$

Acceptance $\approx 72\%$, $\Delta H$ near 0, ESS comfortably above 400.

---

## How That Went

End-to-end pipeline works. Sampler is healthy by every diagnostic. But running evaluation over all 100 test engines:

$$\text{MAE} = 413.6 \text{ cycles}, \quad \text{RMSE} = 862.3 \text{ cycles}, \quad \text{bias} = +413.6 \text{ cycles}$$

Every single error is positive. MAE equals bias to one decimal place. That's not noise — that's a structural problem. The model systematically overpredicts RUL for every engine in the fleet, sometimes by thousands of cycles (engine 59: predicted 7,171, true was 114).

### Why the Linear Model Fails

Look at any $s_{11}$ trajectory. It's not a line. It's roughly flat for the first $\sim 60\%$ of engine life with small noise, then bends upward and accelerates sharply toward failure.

Fitting $h(t) = \alpha + \beta t$ to a truncated test trajectory means the sampler is mostly seeing the flat early region. Consequence: when $\beta$ is fit to flat early data, it comes out small. Then:

$$t_{\text{failure}} = \frac{0.79 - \alpha}{\beta}$$

blows up. Small $\beta$ gives huge $t_{\text{failure}}$ gives huge RUL.

| Engine | Situation | Predicted RUL |
|--------|-----------|---------------|
| e59 | $\beta \approx 0$ | 7,171 cycles |
| e73 | $\beta \approx 0$ | 3,378 cycles |
| e60 | $\beta$ small | 1,184 cycles |
| e99 | $\beta$ small | 1,179 cycles |

These aren't sampler failures — the chain converges fine. They're the linear model honestly extrapolating a shallow slope to a far-away threshold. The prior $\mu_\beta = 0.73$ does pull $\beta$ back, but the test data fights it because early-life data really does look flat.

### Summary

> Linear model + truncated trajectory + accelerating real degradation = systematic overestimation every time.

The implementation is correct. The model is wrong.

---

## What We're Changing To

### The Core Fix: Nonlinear Health Curve

Drop the linear assumption. Degradation is not linear in cycle, so stop pretending it is.

**Exponential model (chosen):**

$$h(t) = \alpha + \beta \cdot e^{\gamma t}$$

Three parameters. Captures accelerating degradation directly. Still hand-differentiable so the HMC machinery stays identical.

**Gradients of the log-likelihood** (prior terms omitted for clarity):

Let $r_t = y_t - (\alpha + \beta e^{\gamma t})$ be the residual at cycle $t$. Then:

$$\frac{\partial \log P}{\partial \alpha} = \frac{1}{\sigma^2} \sum_t r_t$$

$$\frac{\partial \log P}{\partial \beta} = \frac{1}{\sigma^2} \sum_t r_t \cdot e^{\gamma t}$$

$$\frac{\partial \log P}{\partial \gamma} = \frac{1}{\sigma^2} \sum_t r_t \cdot \beta t \cdot e^{\gamma t}$$

**Failure time** solved analytically by inverting $h(t_{\text{fail}}) = \tau$:

$$t_{\text{fail}} = \frac{1}{\gamma} \ln\!\left(\frac{\tau - \alpha}{\beta}\right)$$

**Failure threshold as a distribution** rather than a fixed scalar:

$$\tau \sim \mathcal{N}(0.79,\ \sigma_\tau^2)$$

where $\sigma_\tau$ is the empirical standard deviation of $s_{11}^{\text{norm}}$ at failure across the 100 training engines. One threshold sample is drawn per posterior sample of $(\alpha, \beta, \gamma)$, propagating threshold uncertainty into the RUL distribution.

### Other Options Considered

| Model | Health curve | Notes |
|-------|-------------|-------|
| Quadratic | $\alpha + \beta t + \gamma t^2$ | Simpler gradients, $t_{\text{fail}}$ is a quadratic root |
| Piecewise linear | Flat until changepoint $\tau$, then ramp | $\tau$ becomes a sampled parameter |
| Logistic | $L / (1 + e^{-k(t - t_0)})$ | Full S-curve, more parameters |

---

## Summary of Changes

| | Original model | New model |
|---|---|---|
| Health curve | $\alpha + \beta t$ | $\alpha + \beta e^{\gamma t}$ |
| Parameters | 2 — $(\alpha, \beta)$ | 3 — $(\alpha, \beta, \gamma)$ |
| Failure threshold | Fixed scalar $\tau = 0.79$ | $\tau \sim \mathcal{N}(0.79,\ \sigma_\tau^2)$ |
| Expected bias | $+414$ cycles | $\approx 0$ (to be confirmed) |
| HMC machinery | Unchanged | Unchanged |
| Gradients | 2 analytical expressions | 3 analytical expressions |