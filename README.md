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

engine health is modelled as a linear function of cycle $t$:

$$h(t) = \alpha + \beta t$$

where $\alpha$ is starting health and $\beta$ is the degradation rate. sensor readings are noisy observations of the true health:

$$y_t = h(t) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

RUL is the cycle where health hits the failure threshold $\tau$:

$$t_{\text{RUL}} = \frac{\tau - \alpha}{\beta} - t_{\text{last}}$$

we want the full posterior over parameters $\theta = (\alpha, \beta)$ given observed sensor data $D$:

$$P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta)$$

computing this exactly means integrating over all possible parameter values which is analytically intractable. so instead of computing it, we **sample** from it using **Hamiltonian Monte Carlo (HMC)**.

### Why HMC?

standard random-walk samplers (like Metropolis-Hastings) explore the posterior pretty inefficiently — they stumble around without using any info about the shape of the distribution. HMC uses the **gradient of the log-posterior** as a physical force, simulating a ball rolling across the posterior landscape:

$$H(\theta, p) = \underbrace{-\log P(\theta \mid D)}_{U(\theta)\ \text{potential energy}} + \underbrace{\frac{p^2}{2}}_{K(p)\ \text{kinetic energy}}$$

Hamilton's equations of motion give us:

$$\frac{d\theta}{dt} = p \qquad \frac{dp}{dt} = \nabla \log P(\theta \mid D)$$

the gradient acts as a physical force steering the sampler toward high-probability regions. the intractable $P(D)$ cancels exactly in the acceptance ratio:

$$r = \exp(H_{\text{old}} - H_{\text{new}}) = \frac{P(D \mid \theta^*) \cdot P(\theta^*)}{P(D \mid \theta) \cdot P(\theta)}$$

we never compute the normalising constant. every gradient is derived analytically by hand. no autograd, no JAX.

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
train_FD001.txt ──> fit OLS on all 100 engines → priors on (α, β), failure threshold τ, normalisation constants
                                    ↓
test_FD001.txt ───> HMC sampler ──> posterior P(α, β | D) ──> RUL distribution
                                                                      ↓
RUL_FD001.txt ────────────────────────────────────────────> score predictions
```

**Priors** derived from OLS fits across all 100 training engines:

$$\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2), \quad \beta \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$$

**Failure threshold** $\tau$ = mean $s_{11}^{\text{norm}}$ at last cycle across all training engines (fixed value, not a distribution).

---

## Project Structure

```
├── 01_eda.py              — sensor analysis and health proxy selection
├── 02_preprocessing.py    — clean and normalise data for the sampler
├── 03_hmc_sampler.py      — the HMC implementation (core file)
├── 04_diagnostics.py      — trace plots, ESS, ΔH energy diagnostics
├── 05_rul_prediction.py   — posterior samples → RUL distribution
├── 06_evaluation.py       — predicted vs true RUL across all 100 test engines
├── changes.md             — model change log
└── results/
    ├── linear/            — outputs from the linear model
    └── exponential/       — outputs from the exponential model (wip)
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
- [x] Evaluation — all 100 test engines, RMSE, coverage metric
- [ ] Exponential model — replace linear $h(t)$ with $\alpha + \beta e^{\gamma t}$
- [ ] Threshold as distribution — propagate $\tau$ uncertainty into RUL
- [ ] Re-evaluate — compare RMSE and coverage before vs after

---

## Update

LINEAR MODEL DOO DOO SWITCHING TO NON-LINEAR - SEE CHANGES.MD
