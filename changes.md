# changes.md - pivoting the model

## the original plan

build a from-scratch HMC sampler that infers a posterior over engine degradation parameters and turns that posterior into a full probability distribution over Remaining Useful Life (RUL). numpy only, no PyMC / Stan / autograd.

simplest model that could possibly work:

```
h(t) = α + βt           # linear health curve
y_t  = h(t) + ε         # noisy sensor reading
ε    ~ N(0, σ²)
```

priors gaussian on (α, β), failure declared when `h(t) = threshold`. RUL follows by just inverting that:

```
t_failure = (threshold - α) / β
RUL       = (t_failure - t_last) * global_max_cycle
```

sensor: s11 (highest correlation with cycle across the training fleet, lowest residual std). normalisation:
- s11 normalised to [0,1] using training global min/max
- cycles normalised by training global_max_cycle = 362

priors derived from OLS on all 100 training engines in normalised coords:
- `μ_α ≈ 0.21,  σ_α ≈ 0.15`
- `μ_β ≈ 0.73,  σ_β ≈ 0.24`
- `σ_noise = 0.15` (fixed)

failure threshold: mean s11_norm at the last cycle across all training engines, which comes out to about **0.79**. not 1.0 because most engines die before s11 hits the global max.

hmc tuned on training engine 10:
- `ε = 0.015,  L = 20,  n_samples = 10000,  burn_in = 1000`
- acceptance ~72%, ΔH near 0, ESS comfortably above 400

## how that went

end-to-end pipeline works. sampler is healthy by every diagnostic. but when you run 06_evaluation.py over all 100 test engines:

```
mae:   413.6 cycles
rmse:  862.3 cycles
mean error (bias): +413.6 cycles
```

every single error is positive. mae equals bias to one decimal place. thats not noise, thats a structural problem. the model is systematically overpredicting RUL for every engine in the fleet, sometimes by 7000+ cycles (engine 59 predicted 7171, true was 114).

### why the linear model fails

look at any s11 trajectory. its not a line. its roughly flat for the first ~60% of life with small noise, then bends upward and accelerates sharply toward failure. fitting `h(t) = α + βt` to a truncated test trajectory means we're mostly seeing the flat region.

consequence: when β is fit to flat early data, it comes out small. then:

```
t_failure = (0.79 - α) / β
```

blows up. small β = huge predicted t_failure = huge RUL. you can see it directly:

| engine | situation | predicted RUL |
|--------|-----------|---------------|
| e59    | β ≈ 0     | 7171          |
| e73    | β ≈ 0     | 3378          |
| e60    | β small   | 1184          |
| e99    | β small   | 1179          |

these arent sampler failures, the chain converges fine. they're the linear model honestly extrapolating a shallow slope to a far-away threshold. the prior μ_β = 0.73 does pull β back but the test data is fighting it because early-life data really does look flat.

### tl;dr

> linear model + truncated trajectory + accelerating real degradation = systematic overestimation every time.

the implementation is correct. the model is wrong.

## what we're changing to

### the core fix: nonlinear health curve

drop the linear assumption. degradation is not linear in cycle, so stop pretending it is. options:

**exponential (going with this one):**
```
h(t) = α + β · exp(γt)
```
three parameters, captures accelerating degradation directly, still hand-differentiable so the HMC machinery stays the same.

gradients (prior terms dropped for clarity):
```
∂L/∂α = (1/σ²) Σ r_t
∂L/∂β = (1/σ²) Σ r_t · exp(γt)
∂L/∂γ = (1/σ²) Σ r_t · β · t · exp(γt)
```
where `r_t = y_t - (α + β · exp(γt))`.

failure time solved analytically by rearranging h(t) = threshold:
```
t_fail = (1/γ) · log((threshold - α) / β)
```

other options we might come back to:
- **quadratic**: `h(t) = α + βt + γt²`, even simpler gradients, `t_fail` is a quadratic root
- **piecewise linear**: flat baseline until some change-point τ, then a ramp. τ becomes a sampled parameter
- **logistic**: `h(t) = L / (1 + exp(-k(t - t₀)))`, full S-curve, more parameters



| | original | new |
|---|---|---|
| health curve | `α + βt` | `α + β·exp(γt)` |
| parameters | 2 | 3 |
| failure threshold | scalar 0.79 | `N(0.79, σ_thr²)` |
| RUL output | point + narrow CI | full posterior distribution |
| expected bias | +414 cycles | ≈ 0 (TBD) |

the math, priors, HMC, leapfrog, diagnostics all stay. what changes is the function we're fitting to the data and how we treat the threshold.
