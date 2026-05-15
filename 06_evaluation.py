import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = "results/fleet_S11/figures"
os.makedirs(fig_dir, exist_ok=True)

# -load data once
train_data = np.loadtxt("CMAPSSData/train_FD001.txt")
test_data  = np.loadtxt("CMAPSSData/test_FD001.txt")
true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")

s11_col = 15
s11_min = train_data[:, s11_col].min()
s11_max = train_data[:, s11_col].max()
global_max_cycle = train_data[:, 1].max()

#  derive priors from OLS across all 100 training engines 
# fit h(t) = alpha + beta*t to each training engine in global cycle coords
# then use the distribution of (alpha, beta) across engines as our prior
ols_alphas, ols_betas = [], []
for eid in range(1, 101):
    e = train_data[train_data[:, 0] == eid]
    t = e[:, 1] / global_max_cycle
    y = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    A = np.stack([np.ones_like(t), t], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    ols_alphas.append(a)
    ols_betas.append(b)

ols_alphas = np.array(ols_alphas)
ols_betas  = np.array(ols_betas)

mu_alpha    = ols_alphas.mean()       # 0.21
sigma_alpha = ols_alphas.std() + 0.05  # add small buffer above empirical std
mu_beta     = ols_betas.mean()        # 0.73
sigma_beta  = ols_betas.std()  + 0.05  # add small buffer above empirical std
sigma_noise = 0.15

print(f"derived priors — mu_alpha={mu_alpha:.3f}  sigma_alpha={sigma_alpha:.3f}  "
      f"mu_beta={mu_beta:.3f}  sigma_beta={sigma_beta:.3f}")

# failure threshold: mean s11_norm at last cycle across all training engines
# training engines run to actual failure, so the last s11_norm is where they died
# use the mean as our best estimate of the failure threshold
end_s11 = []
for eid in range(1, 101):
    e = train_data[train_data[:, 0] == eid]
    s11_norm = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    end_s11.append(s11_norm[-1])

failure_threshold = np.mean(end_s11)
print(f"failure threshold (mean s11_norm at failure): {failure_threshold:.4f}")

# hmc hyperparameters — tuned on training engine 10, 71.98% acceptance
epsilon   = 0.015
L         = 20
n_samples = 10000
burn_in   = 1000


def log_posterior(alpha, beta, cycles, y_obs):
    # log likelihood: sum of log N(y_t | alpha + beta*t, sigma)
    ll = -0.5 * np.sum(((y_obs - (alpha + beta * cycles)) / sigma_noise) ** 2)
    # log prior: gaussian on alpha and beta
    lp = -0.5 * ((alpha - mu_alpha) / sigma_alpha) ** 2 \
       - 0.5 * ((beta  - mu_beta)  / sigma_beta)  ** 2
    return ll + lp


def compute_gradients(alpha, beta, cycles, y_obs):
    # analytical gradients of log_posterior — drives the leapfrog integrator
    r = y_obs - (alpha + beta * cycles)
    ga = (1 / sigma_noise**2) * np.sum(r)          - (alpha - mu_alpha) / sigma_alpha**2
    gb = (1 / sigma_noise**2) * np.sum(cycles * r) - (beta  - mu_beta)  / sigma_beta**2
    return np.array([ga, gb])


def leapfrog(theta, p, cycles, y_obs):
    # staggered half-kick / full-move / half-kick — second-order energy conserving
    theta = theta.copy(); p = p.copy()
    p += (epsilon / 2) * compute_gradients(theta[0], theta[1], cycles, y_obs)
    for _ in range(L - 1):
        theta += epsilon * p
        p     += epsilon * compute_gradients(theta[0], theta[1], cycles, y_obs)
    theta += epsilon * p
    p     += (epsilon / 2) * compute_gradients(theta[0], theta[1], cycles, y_obs)
    return theta, p


def run_hmc(cycles, y_obs):
    # hmc sampler — returns (samples, acceptance_rate)
    theta = np.array([mu_alpha, mu_beta])
    samples  = np.zeros((n_samples, 2))
    accepted = 0

    for i in range(n_samples):
        p = np.random.randn(2)
        H_cur = -log_posterior(theta[0], theta[1], cycles, y_obs) + 0.5 * np.dot(p, p)
        theta_prop, p_prop = leapfrog(theta, p, cycles, y_obs)
        H_prop = -log_posterior(theta_prop[0], theta_prop[1], cycles, y_obs) + 0.5 * np.dot(p_prop, p_prop)

        if np.log(np.random.rand()) < H_cur - H_prop:
            theta = theta_prop
            accepted += 1
        samples[i] = theta

    return samples, accepted / n_samples


# --- run over all 100 test engines ---
results = []

for engine_id in range(1, 101):
    # preprocess
    e = test_data[test_data[:, 0] == engine_id]
    cycles_raw = e[:, 1]
    s11_raw    = e[:, s11_col]

    cycles  = cycles_raw / global_max_cycle
    y_obs   = (s11_raw - s11_min) / (s11_max - s11_min)
    t_last  = cycles_raw.max() / global_max_cycle
    true_rul = true_rul_all[engine_id - 1]

    # sample
    samples, acc = run_hmc(cycles, y_obs)
    post    = samples[burn_in:]
    alpha_s = post[:, 0]
    beta_s  = post[:, 1]

    # rul — how many real cycles until h(t) = failure_threshold
    rul_norm = (failure_threshold - alpha_s) / beta_s
    rul_real = (rul_norm - t_last) * global_max_cycle
    valid    = rul_real > 0
    rul_real = rul_real[valid]

    if len(rul_real) == 0:
        mean_rul = 0.0
        ci_low = ci_high = 0.0
    else:
        mean_rul = rul_real.mean()
        ci_low   = np.percentile(rul_real, 5)
        ci_high  = np.percentile(rul_real, 95)

    error = mean_rul - true_rul
    results.append({
        "engine_id": engine_id,
        "mean_rul":  mean_rul,
        "ci_low":    ci_low,
        "ci_high":   ci_high,
        "true_rul":  true_rul,
        "error":     error,
        "acc":       acc,
        "n_cycles":  len(cycles_raw),
    })

    print(f"e{engine_id:3d}  cycles={len(cycles_raw):3d}  acc={acc:.0%}  "
          f"pred={mean_rul:6.1f}  90%CI<={ci_high:6.0f}  true={true_rul:5.0f}  err={error:+.1f}")

# --- summary stats ---
errors     = np.array([r["error"]     for r in results])
abs_errors = np.abs(errors)
true_ruls  = np.array([r["true_rul"]  for r in results])
pred_ruls  = np.array([r["mean_rul"]  for r in results])

ci_highs = np.array([r["ci_high"] for r in results])
print(f"\n--- fleet summary ---")
print(f"mae:   {abs_errors.mean():.1f} cycles")
print(f"rmse:  {np.sqrt((errors**2).mean()):.1f} cycles")
print(f"mean error (bias): {errors.mean():+.1f} cycles")
print(f"median '90% fails within': {np.median(ci_highs):.0f} cycles  (vs median true RUL: {np.median(true_ruls):.0f})")

# --- plot 1: predicted vs true RUL ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(true_ruls, pred_ruls, alpha=0.6, s=30, color="steelblue")
lims = [0, max(true_ruls.max(), pred_ruls.max()) * 1.05]
ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
ax.set_xlabel("true RUL (cycles)")
ax.set_ylabel("predicted RUL (cycles)")
ax.set_title("predicted vs true RUL — all 100 test engines, s11")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{fig_dir}/pred_vs_true_rul.png", dpi=120)
plt.close()
print(f"saved pred_vs_true_rul.png")

# --- plot 2: error distribution ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(errors, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(0,            color="red",    linewidth=1.2, linestyle="--", label="zero error")
ax.axvline(errors.mean(), color="orange", linewidth=1.2, linestyle="--", label=f"mean={errors.mean():+.1f}")
ax.set_xlabel("prediction error (pred - true) cycles")
ax.set_ylabel("count")
ax.set_title("RUL prediction error distribution — 100 test engines")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{fig_dir}/error_distribution.png", dpi=120)
plt.close()
print(f"saved error_distribution.png")
