# Bayesian RUL Prediction with Hamiltonian Monte Carlo

Predicting when a turbofan engine will fail — not as a single number, but as a full probability distribution. Built from scratch using only NumPy.

---

## The Problem

Most predictive maintenance models give you a point estimate: *"this engine will fail in 47 cycles."* That single number throws away all the uncertainty in the prediction. In a real maintenance setting, the difference between 90% confident and 50% confident matters enormously — one means you can wait, the other means you probably shouldn't.

This project produces outputs like: *"90% confident this engine fails before cycle 215"* — a full distribution over Remaining Useful Life (RUL), with honest uncertainty baked in.

---

## The Approach

We use **Bayesian inference** to infer a posterior distribution over engine degradation parameters, then propagate that uncertainty through to a distribution over RUL.

Engine health is modelled as a linear function of cycle:

```
h(t) = α + βt
```

where `α` is starting health and `β` is the degradation rate (negative — engines get worse over time). Sensor readings are noisy observations of true health:

```
y_t = h(t) + ε,    ε ~ N(0, σ²)
```

RUL is the cycle where health reaches zero: `t_RUL = -α/β`

We want the full posterior over the parameters `(α, β)` given the observed sensor data `D`:

```
P(α, β | D) ∝ P(D | α, β) · P(α, β)
```

Computing this exactly requires integrating over all possible parameter values — analytically intractable. So instead of computing it, we **sample** from it using **Hamiltonian Monte Carlo (HMC)**.

### Why HMC?

Standard random-walk samplers (like Metropolis-Hastings) explore the posterior inefficiently — they stumble around without using any information about the shape of the distribution. HMC uses the **gradient of the log-posterior** as a physical force, simulating a ball rolling across the posterior landscape. This lets it traverse the space much more efficiently, especially as the number of parameters grows.

Every gradient is derived analytically by hand. No autograd, no JAX.

---

## The Dataset

**NASA C-MAPSS FD001** — turbofan engine degradation simulation data from the NASA Prognostics Center of Excellence.

- 100 engines run to failure
- 26 columns: engine ID, cycle number, 3 operating settings, 21 sensor measurements
- FD001 specifically has one operating condition and one fault mode (HPC degradation), making it the cleanest sub-dataset to start with

The dataset is not included in this repo. It can be downloaded from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

Place the files in `CMAPSSData/`:
```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

---

## Project Structure

```
├── 01_eda.py              — sensor analysis and health proxy selection
├── 02_preprocessing.py    — clean and normalise data for the sampler
├── 03_hmc_sampler.py      — the HMC implementation (core file)
├── 04_diagnostics.py      — trace plots, R-hat, ESS, energy diagnostics
├── 05_rul_prediction.py   — posterior samples → RUL distribution
├── 06_evaluation.py       — predicted vs true RUL across all test engines
└── results/
    ├── figures/           — all plots
    └── samples/           — saved posterior samples (.npy)
```

---

## Where We Are

- [x] EDA — loaded dataset, computed per-sensor correlation with cycle, selected `s11` as the health proxy (highest degradation signal, |corr| = 0.634)
- [ ] Preprocessing — normalise s11, prepare clean arrays for the sampler
- [ ] HMC sampler — implement leapfrog, acceptance step, full sampler loop
- [ ] Diagnostics — verify chain health before trusting any samples
- [ ] RUL prediction — turn posterior samples into a RUL distribution
- [ ] Evaluation — test against all 100 engines, compute RMSE

---

## Constraints

- NumPy only for the sampler — no PyMC, Stan, or probabilistic programming libraries
- No automatic differentiation — all gradients derived and implemented by hand
- Naive implementation first, optimisations later
