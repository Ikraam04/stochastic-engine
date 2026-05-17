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

## How That Went (Linear Model Results)

End-to-end pipeline works. Sampler is healthy by every diagnostic. But running evaluation over all 100 test engines:

$$\text{MAE} = 314.7 \text{ cycles}, \quad \text{RMSE} = 371.4 \text{ cycles}, \quad \text{bias} = +314.7 \text{ cycles}$$

$$\text{coverage (90\% CI)} = 4\%$$

every single error is positive. MAE equals bias to one decimal place. thats not noise — thats a structural problem. the model systematically overpredicts RUL for every engine in the fleet.

the 4% coverage is the clearest signal. a 90% CI should contain the true value ~90% of the time. 4% doesnt mean the intervals are too narrow — it means theyre in completely the wrong place.

### Why the Linear Model Fails

look at any $s_{11}$ trajectory. its not a line. its roughly flat for the first ~60% of engine life, then bends upward and accelerates sharply toward failure.

fitting $h(t) = \alpha + \beta t$ to a truncated test trajectory means the sampler is mostly seeing the flat early region. when $\beta$ is fit to flat early data, it comes out small. then:

$$t_{\text{failure}} = \frac{0.79 - \alpha}{\beta}$$

blows up. small $\beta$ gives huge $t_{\text{failure}}$ gives huge RUL.

| Engine | Situation | Predicted RUL |
|--------|-----------|---------------|
| e59 | $\beta \approx 0$ | 7,171 cycles |
| e73 | $\beta \approx 0$ | 3,378 cycles |
| e60 | $\beta$ small | 1,184 cycles |
| e99 | $\beta$ small | 1,179 cycles |

these arent sampler failures — the chain converges fine. theyre the linear model honestly extrapolating a shallow slope to a far-away threshold. the prior $\mu_\beta = 0.73$ does pull $\beta$ back, but the test data fights it because early-life data really does look flat.

### Summary

> linear model + truncated trajectory + accelerating real degradation = systematic overestimation every time.

the implementation is correct. the model is wrong.

---

## What We Changed To

### The Core Fix: Nonlinear Health Curve

drop the linear assumption. degradation is not linear in cycle, so stop pretending it is.

**Exponential model (chosen):**

$$h(t) = \alpha + \beta \cdot e^{\gamma t}$$

three parameters. captures accelerating degradation directly. still hand-differentiable so the HMC machinery stays identical.

**Gradients of the log-likelihood** (prior terms omitted for clarity):

let $r_t = y_t - (\alpha + \beta e^{\gamma t})$ be the residual at cycle $t$. then:

$$\frac{\partial \log P}{\partial \alpha} = \frac{1}{\sigma^2} \sum_t r_t$$

$$\frac{\partial \log P}{\partial \beta} = \frac{1}{\sigma^2} \sum_t r_t \cdot e^{\gamma t}$$

$$\frac{\partial \log P}{\partial \gamma} = \frac{1}{\sigma^2} \sum_t r_t \cdot \beta t \cdot e^{\gamma t}$$

**Failure time** solved analytically by inverting $h(t_{\text{fail}}) = \tau$:

$$t_{\text{fail}} = \frac{1}{\gamma} \ln\!\left(\frac{\tau - \alpha}{\beta}\right)$$

### Other Options Considered

| Model | Health curve | Notes |
|-------|-------------|-------|
| Quadratic | $\alpha + \beta t + \gamma t^2$ | Simpler gradients, $t_{\text{fail}}$ is a quadratic root |
| Piecewise linear | Flat until changepoint, then ramp | changepoint becomes a sampled parameter |
| Logistic | $L / (1 + e^{-k(t - t_0)})$ | Full S-curve, more parameters |

---

## Exponential Model — what we built

### getting priors for γ

before sampling we need to know what γ looks like across the fleet. fit $h(t) = \alpha + \beta e^{\gamma t}$ to all 100 training engines using `scipy.optimize.curve_fit`. clip the top/bottom 5% of fits before computing mean/std (a few engines dont converge cleanly). results:

$$\mu_\alpha \approx 0.287, \quad \sigma_\alpha \approx 0.078$$

$$\mu_\beta \approx 0.0116, \quad \sigma_\beta \approx 0.0051$$

$$\mu_\gamma \approx 7.16, \quad \sigma_\gamma \approx 1.58$$

### reparametrisation

sampling $(\alpha, \beta, \gamma)$ directly doesnt work well. $\beta$ and $\gamma$ live on very different scales ($\beta \sim 0.01$, $\gamma \sim 7$), and a single leapfrog step size $\varepsilon$ cant handle all three well at once. on top of that, $\beta > 0$ and $\gamma > 0$ are hard constraints that need to be enforced somehow.

fix: sample in log space instead. define:

$$\phi = \log\beta, \quad \psi = \log\gamma$$

now $\beta = e^\phi > 0$ and $\gamma = e^\psi > 0$ hold by construction — `exp()` is always positive so the constraint is automatic. the three params are also closer in scale, so one $\varepsilon$ actually works consistently.

the priors on $\phi$ and $\psi$ come from the delta method approximation. if X is approximately normal with mean $\mu$ and std $\sigma$, then $\log(X)$ is approximately normal with:

$$\mu_\phi = \log(\mu_\beta), \quad \sigma_\phi \approx \frac{\sigma_\beta}{\mu_\beta}$$

this is a first-order Taylor expansion of $\log(\cdot)$ around the mean. same for $\psi$.

### gradients in reparametrised space

chain rule. $\beta$ and $\gamma$ are now functions of $\phi$ and $\psi$, so:

$$\frac{\partial \log P}{\partial \phi} = \frac{\partial \log P}{\partial \beta} \cdot \underbrace{\frac{d\beta}{d\phi}}_{\beta} = \frac{\partial \log P}{\partial \beta} \cdot \beta$$

$$\frac{\partial \log P}{\partial \psi} = \frac{\partial \log P}{\partial \gamma} \cdot \gamma$$

the original gradient expressions just get multiplied by $\beta$ and $\gamma$ respectively. written out in full:

$$\frac{\partial \log P}{\partial \phi} = \frac{1}{\sigma^2} \sum_t r_t \cdot e^{\gamma t} \cdot \beta \;-\; \frac{\phi - \mu_\phi}{\sigma_\phi^2}$$

$$\frac{\partial \log P}{\partial \psi} = \frac{1}{\sigma^2} \sum_t r_t \cdot \beta t \cdot e^{\gamma t} \cdot \gamma \;-\; \frac{\psi - \mu_\psi}{\sigma_\psi^2}$$

### gradient clipping

4 engines (E49, E62, E91, E93 — all 230+ cycles) had 0% acceptance on the first fleet run. the problem is $e^{\gamma t}$ gets very large for long trajectories near failure. this pushes gradient magnitudes to huge values and leapfrog integration diverges.

fix: clip each gradient component to $\pm 50$ before every leapfrog half-step:

```python
return np.clip(np.array([g_alpha, g_phi, g_psi]), -50, 50)
```

this doesnt change the target distribution. the posterior we're sampling from is still exactly $P(\alpha, \phi, \psi \mid D)$. it just stabilises the numerical integrator on very steep parts of the posterior landscape. all 4 engines produced valid samples after this fix.

this is different from per-engine $\varepsilon$ tuning, which would be fiddling with the model for each engine. gradient clipping is a property of the integrator, not the model. real tools like Stan/PyMC handle this automatically with NUTS — it adapts $\varepsilon$ during warmup so the integrator never sees exploding gradients in the first place.

### hyperparameters

tuned on engine 10. after reparametrisation, $\varepsilon = 0.015$ gives 78% acceptance consistently across runs. without reparametrisation the results varied a lot run-to-run for the same $\varepsilon$ because the posterior was badly conditioned.

$$\varepsilon = 0.015, \quad L = 15, \quad N = 10{,}000, \quad N_{\text{burn}} = 1{,}000$$

### failure threshold

kept fixed: $\tau \approx 0.7915$ (mean normalised $s_{11}$ at last cycle across training engines). we planned to model this as a distribution to propagate threshold uncertainty into the RUL estimate, but didnt end up needing it — the exponential model already gets 94% coverage without it.

### RUL formula

samples are stored as $(\alpha, \phi, \psi)$ in log space. to compute RUL, first recover:

$$\beta = e^\phi, \quad \gamma = e^\psi$$

then invert $h(t_{\text{fail}}) = \tau$:

$$t_{\text{fail}} = \frac{1}{\gamma} \ln\!\left(\frac{\tau - \alpha}{\beta}\right)$$

$$\text{RUL} = (t_{\text{fail}} - t_{\text{last}}) \times t_{\text{max}}$$

only valid when $\tau - \alpha > 0$ (engine hasnt already overshot the threshold in this posterior sample). samples giving RUL $\leq 0$ are discarded.

### results

fleet eval over all 100 test engines using `multiprocessing.Pool` (one HMC chain per engine, 10k samples, 1k burn-in):

| metric | linear model | exponential model |
|--------|-------------|-------------------|
| MAE | 314.7 cycles | **15.2 cycles** |
| RMSE | 371.4 cycles | **20.1 cycles** |
| bias | +314.7 cycles | **+6.7 cycles** |
| coverage (90% CI) | 4% | **94%** |

**coverage** = fraction of engines where the true RUL fell inside our [5th, 95th] percentile credible interval. for a perfectly calibrated 90% CI this should be ~90%.

4% for linear: the CIs are in the wrong place due to model misspecification. the sampler is working correctly — the model shape is just wrong. the posterior is precisely wrong.

94% for exponential: slightly conservative (intervals a bit wider than strictly needed) but well-calibrated. the extra 4% over the ideal 90% means we're hedging a little more than the data requires, which is fine for a safety-critical application.

MAE of 15.2 cycles is competitive with LSTM-based approaches on CMAPSS FD001 (typical reported MAE: 12-18 cycles). this model also gives a full uncertainty distribution — neural networks dont give you that.
