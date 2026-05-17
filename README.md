# Bayesian RUL Prediction with Hamiltonian Monte Carlo

predicting when a turbofan engine will fail, not as a single number but as a full probability distribution. built from scratch using only NumPy.

---

## The Problem

most predictive maintenance models give you a point estimate: *"this engine will fail in 47 cycles."* that single number throws away all the uncertainty in the prediction. in a real maintenance setting, the difference between 90% confident and 50% confident matters a lot — one means you can wait, the other means you probably shouldn't.

this project gives outputs like: *"90% confident this engine fails before cycle 215"* — a full distribution over Remaining Useful Life (RUL), with actual uncertainty baked in.

---

## The Approach

we use **Bayesian inference** to get a posterior distribution over engine degradation parameters, then push that uncertainty through to a distribution over RUL.

### The Model

engine health follows an exponential degradation curve:

$$h(t) = \alpha + \beta \cdot e^{\gamma t}$$

where $\alpha$ is baseline health, $\beta$ is the degradation magnitude, and $\gamma$ controls how fast it accelerates. sensor readings are noisy observations:

$$y_t = h(t) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

RUL is the cycle where health hits the failure threshold $\tau$, solved analytically by inverting $h(t_{\text{fail}}) = \tau$:

$$t_{\text{RUL}} = \frac{1}{\gamma} \ln\!\left(\frac{\tau - \alpha}{\beta}\right) - t_{\text{last}}$$

we want the full posterior over parameters $\theta = (\alpha, \beta, \gamma)$ given observed sensor data $D$:

$$P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta)$$

computing this exactly means integrating over all possible parameter values, which is analytically intractable. so instead we **sample** from it using **Hamiltonian Monte Carlo (HMC)**.

> **why not linear?** we first tried $h(t) = \alpha + \beta t$. the sampler worked fine but MAE was 315 cycles with a systematic +315 bias — the model extrapolates a flat early-life slope all the way to the threshold, overshooting by hundreds of cycles every time. see [changes.md](changes.md) for the full breakdown.

### Reparametrisation

we dont sample $\beta$ and $\gamma$ directly. instead we sample:

$$\phi = \log\beta, \quad \psi = \log\gamma$$

so $\beta = e^\phi > 0$ and $\gamma = e^\psi > 0$ are guaranteed by construction. it also puts the three params on a more similar scale, which means one step size $\varepsilon$ works for all three. the chain rule gives the gradient in reparametrised space:

$$\frac{\partial \log P}{\partial \phi} = \frac{\partial \log P}{\partial \beta} \cdot \beta, \qquad \frac{\partial \log P}{\partial \psi} = \frac{\partial \log P}{\partial \gamma} \cdot \gamma$$

every gradient is derived analytically by hand. no autograd, no JAX.

### Why HMC?

standard random-walk samplers (like Metropolis-Hastings) explore the posterior inefficiently — they stumble around without using any info about the shape of the distribution. HMC uses the **gradient of the log-posterior** as a physical force, simulating a ball rolling across the posterior landscape:

$$H(\theta, p) = \underbrace{-\log P(\theta \mid D)}_{U(\theta)\ \text{potential energy}} + \underbrace{\frac{p^2}{2}}_{K(p)\ \text{kinetic energy}}$$

Hamilton's equations of motion give us:

$$\frac{d\theta}{dt} = p \qquad \frac{dp}{dt} = \nabla \log P(\theta \mid D)$$

the gradient acts as a physical force steering the sampler toward high-probability regions. the intractable $P(D)$ cancels exactly in the acceptance ratio — we never compute the normalising constant:
$$
r = \exp(H_{\text{old}} - H_{\text{new}})
= \frac{P(D \mid \theta_{\text{new}})\cdot P(\theta_{\text{new}})} {P(D \mid \theta)\cdot P(\theta)}
$$

---

## The Dataset

**NASA C-MAPSS FD001** — turbofan engine degradation simulation from the NASA Prognostics Center of Excellence.

- 100 training engines run to failure, 100 test engines truncated before failure
- 26 columns: engine ID, cycle number, 3 operating settings, 21 sensor measurements
- FD001: one operating condition, one fault mode (HPC degradation) — the cleanest sub-dataset

download from the [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) and place files in `CMAPSSData/`:

```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

**Sensor selection:** $s_{11}$ (column index 15) selected as the health proxy. highest degradation signal across the training fleet ($|\rho| = 0.634$), clear monotonic trend across engine life.

**Normalisation:**
- $s_{11}$ normalised to $[0, 1]$ using global training min/max
- cycles normalised by $t_{\text{max}} = 362$ (global max cycle across training fleet)

---

## Pipeline

```
train_FD001.txt ──> curve_fit exp model on all 100 engines
                    → priors on (α, β, γ), failure threshold τ, normalisation constants
                                    ↓
test_FD001.txt ───> HMC sampler (reparametrised: φ=log β, ψ=log γ)
                    → posterior P(α, φ, ψ | D) → RUL distribution
                                                        ↓
RUL_FD001.txt ──────────────────────────────> score predictions
```

**Priors** for $(\alpha, \beta, \gamma)$ derived by fitting the exponential model to all 100 training engines with `scipy.optimize.curve_fit`, then clipping outliers (5th/95th percentile) before computing mean/std. priors on $\phi = \log\beta$ and $\psi = \log\gamma$ use the delta method:

$$\mu_\phi = \log(\mu_\beta), \quad \sigma_\phi \approx \frac{\sigma_\beta}{\mu_\beta}$$

**Failure threshold** $\tau = 0.7915$ = mean $s_{11}^{\text{norm}}$ at last cycle across all training engines (fixed scalar, not a distribution).

---

## Results

fleet eval over all 100 test engines. one HMC chain per engine, 10k samples, 1k burn-in.

| metric | linear $h = \alpha + \beta t$ | exponential $h = \alpha + \beta e^{\gamma t}$ |
|--------|-------------------------------|-----------------------------------------------|
| MAE | 314.7 cycles | **15.2 cycles** |
| RMSE | 371.4 cycles | **20.1 cycles** |
| bias | +314.7 cycles | **+6.7 cycles** |
| coverage (90% CI) | 4% | **94%** |

**coverage** = fraction of true RULs that fall inside the [5th, 95th] percentile credible interval. for a perfectly calibrated 90% CI this should be ~90%.

4% for linear: the CIs are in the wrong place due to model misspecification — the sampler is working correctly, the model shape is just wrong.

94% for exponential: slightly conservative (intervals a touch wider than needed) but well-calibrated. good for a safety-critical application.

MAE of 15.2 cycles is competitive with LSTM benchmarks on CMAPSS FD001 (typical range: 12–18 cycles), and this model also gives full uncertainty quantification.

---

## Project Structure

```
├── fit_exp_priors.py        — fit exp curves to training engines, derive priors for γ
├── 01_eda.py                — sensor analysis and health proxy selection
├── 02_preprocessing.py      — clean and normalise data for the sampler
├── 03_hmc_sampler.py        — linear model HMC (historical)
├── 03_hmc_sampler_exp.py    — exponential model HMC with reparametrisation
├── 04_diagnostics.py        — linear model diagnostics
├── 04_diagnostics_exp.py    — trace plots, ESS, ΔH, autocorrelation (exp model)
├── 05_rul_prediction.py     — linear model RUL prediction
├── 05_rul_prediction_exp.py — posterior samples → RUL distribution (exp model)
├── 06_evaluation.py         — fleet eval, linear model
├── 06_evaluation_exp.py     — fleet eval, exponential model
├── changes.md               — model change log with full math
└── results/
    ├── linear/              — outputs from the linear model
    └── exponential/         — outputs from the exponential model
```

---

## Constraints

- NumPy only for the sampler — no PyMC, Stan, or any PPL
- no automatic differentiation — all gradients derived and coded by hand
- naive implementation first, optimisations later

---

## Progress

- [x] EDA — sensor correlation analysis, $s_{11}$ selected as health proxy
- [x] Preprocessing — global normalisation, clean arrays for sampler
- [x] HMC sampler — leapfrog integrator, accept/reject, full sampler loop
- [x] Diagnostics — trace plots, ESS, $\Delta H$ distribution, autocorrelation
- [x] RUL prediction — posterior samples → RUL distribution with credible intervals
- [x] Evaluation — all 100 test engines, MAE/RMSE/bias/coverage
- [x] Exponential model — reparametrised HMC ($\phi = \log\beta$, $\psi = \log\gamma$), gradient clipping, MAE=15.2
