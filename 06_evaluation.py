import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

model   = "linear"
fig_dir = f"results/{model}/fleet_S11/figures"
os.makedirs(fig_dir, exist_ok=True)

# load data once — workers get their slice passed in as arrays, not the full file
train_data   = np.loadtxt("CMAPSSData/train_FD001.txt")
test_data    = np.loadtxt("CMAPSSData/test_FD001.txt")
true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")

s11_col          = 15
s11_min          = train_data[:, s11_col].min()
s11_max          = train_data[:, s11_col].max()
global_max_cycle = train_data[:, 1].max()

# derive priors from OLS across all 100 training engines
# fit h(t) = alpha + beta*t in normalised global cycle coords
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

mu_alpha    = ols_alphas.mean()
sigma_alpha = ols_alphas.std() + 0.05
mu_beta     = ols_betas.mean()
sigma_beta  = ols_betas.std() + 0.05
sigma_noise = 0.15

# failure threshold: mean s11_norm at last cycle across all training engines
end_s11 = []
for eid in range(1, 101):
    e = train_data[train_data[:, 0] == eid]
    s11_norm = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    end_s11.append(s11_norm[-1])
failure_threshold = np.mean(end_s11)

# hmc hyperparameters — tuned on training engine 10, 71.98% acceptance
epsilon   = 0.015
L         = 20
n_samples = 10000
burn_in   = 1000


def process_engine(args):
    # same RUL computation as 05_rul_prediction.py — just repeated for every engine in the fleet
    # each worker gets its own engine's data slice + the shared config dict
    engine_id, cycles, y_obs, t_last, true_rul, cfg = args

    mu_a    = cfg["mu_alpha"];    sig_a  = cfg["sigma_alpha"]
    mu_b    = cfg["mu_beta"];     sig_b  = cfg["sigma_beta"]
    sig_n   = cfg["sigma_noise"]
    eps     = cfg["epsilon"];     Lsteps = cfg["L"]
    ns      = cfg["n_samples"];   bi     = cfg["burn_in"]
    thresh  = cfg["failure_threshold"]
    gmc     = cfg["global_max_cycle"]

    def log_posterior(alpha, beta):
        ll = -0.5 * np.sum(((y_obs - (alpha + beta * cycles)) / sig_n) ** 2)
        lp = -0.5 * ((alpha - mu_a) / sig_a) ** 2 \
           - 0.5 * ((beta  - mu_b) / sig_b)  ** 2
        return ll + lp

    def compute_gradients(alpha, beta):
        r  = y_obs - (alpha + beta * cycles)
        ga = (1 / sig_n**2) * np.sum(r)          - (alpha - mu_a) / sig_a**2
        gb = (1 / sig_n**2) * np.sum(cycles * r) - (beta  - mu_b) / sig_b**2
        return np.array([ga, gb])

    def leapfrog(theta, p):
        theta = theta.copy(); p = p.copy()
        p += (eps / 2) * compute_gradients(theta[0], theta[1])
        for _ in range(Lsteps - 1):
            theta += eps * p
            p     += eps * compute_gradients(theta[0], theta[1])
        theta += eps * p
        p     += (eps / 2) * compute_gradients(theta[0], theta[1])
        return theta, p

    theta    = np.array([mu_a, mu_b])
    samples  = np.zeros((ns, 2))
    accepted = 0

    for i in range(ns):
        p      = np.random.randn(2)
        H_cur  = -log_posterior(theta[0], theta[1]) + 0.5 * np.dot(p, p)
        th_p, p_p = leapfrog(theta, p)
        H_prop = -log_posterior(th_p[0], th_p[1]) + 0.5 * np.dot(p_p, p_p)
        if np.log(np.random.rand()) < H_cur - H_prop:
            theta = th_p
            accepted += 1
        samples[i] = theta

    acc     = accepted / ns
    post    = samples[bi:]
    alpha_s = post[:, 0]
    beta_s  = post[:, 1]

    # RUL: solve h(t) = threshold for t, then convert back to real cycles
    rul_norm = (thresh - alpha_s) / beta_s
    rul_real = (rul_norm - t_last) * gmc
    rul_real = rul_real[rul_real > 0]

    if len(rul_real) == 0:
        mean_rul = ci_low = ci_high = 0.0
    else:
        mean_rul = rul_real.mean()
        ci_low   = np.percentile(rul_real, 5)
        ci_high  = np.percentile(rul_real, 95)

    return {
        "engine_id": engine_id,
        "mean_rul":  mean_rul,
        "ci_low":    ci_low,
        "ci_high":   ci_high,
        "true_rul":  true_rul,
        "error":     mean_rul - true_rul,
        "acc":       acc,
        "n_cycles":  len(cycles),
    }


if __name__ == "__main__":
    print(f"priors: mu_alpha={mu_alpha:.3f}  sigma_alpha={sigma_alpha:.3f}  "
          f"mu_beta={mu_beta:.3f}  sigma_beta={sigma_beta:.3f}")
    print(f"failure threshold: {failure_threshold:.4f}")

    cfg = dict(
        mu_alpha=mu_alpha,sigma_alpha=sigma_alpha,
        mu_beta=mu_beta, sigma_beta=sigma_beta,
        sigma_noise=sigma_noise, epsilon=epsilon,
        L=L, n_samples=n_samples, burn_in=burn_in,
        failure_threshold=failure_threshold,
        global_max_cycle=global_max_cycle,
    )

    # build one args tuple per engine — each worker gets its own data slice
    engine_args = []
    for eid in range(1, 101):
        e          = test_data[test_data[:, 0] == eid]
        cycles_raw = e[:, 1]
        cycles     = cycles_raw / global_max_cycle
        y_obs      = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
        t_last     = cycles_raw.max() / global_max_cycle
        true_rul   = true_rul_all[eid - 1]
        engine_args.append((eid, cycles, y_obs, t_last, true_rul, cfg))

    # imap_unordered — prints results as engines finish (out of order is fine)
    results = []
    with Pool() as pool:
        for r in pool.imap_unordered(process_engine, engine_args):
            results.append(r)
            print(f"E{r['engine_id']:3d}: cycles={r['n_cycles']:3d}  acc={r['acc']:4.0%}  "
                  f"{r['ci_low']:.0f}<=x<={r['ci_high']:.0f}  "
                  f"pred={r['mean_rul']:.0f}  true={r['true_rul']:.0f}  err={r['error']:+.0f}")

    results.sort(key=lambda x: x["engine_id"])

    errors     = np.array([r["error"]    for r in results])
    abs_errors = np.abs(errors)
    true_ruls  = np.array([r["true_rul"] for r in results])
    pred_ruls  = np.array([r["mean_rul"] for r in results])
    ci_lows    = np.array([r["ci_low"]   for r in results])
    ci_highs   = np.array([r["ci_high"]  for r in results])

    # coverage = fraction of true RULs that fall inside our 90% CI
    # want this close to 90% — too high means CI is too wide, too low means too confident
    covered  = (true_ruls >= ci_lows) & (true_ruls <= ci_highs)
    coverage = covered.mean()

    print(f"Summary:")
    print(f"mae:   {abs_errors.mean():.1f} cycles")
    print(f"rmse:  {np.sqrt((errors**2).mean()):.1f} cycles")
    print(f"mean error (bias): {errors.mean():+.1f} cycles")
    print(f"coverage (90% CI): {coverage:.1%}  ({covered.sum()}/100 engines)")
    print(f"median: {np.median(ci_highs):.0f} cycles  "
          f"(vs median true RUL: {np.median(true_ruls):.0f})")

    # plot 1: predicted vs true RUL
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


    # plot 2: error distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0,             color="red",    linewidth=1.2, linestyle="--", label="zero error")
    ax.axvline(errors.mean(), color="orange", linewidth=1.2, linestyle="--",
               label=f"mean={errors.mean():+.1f}")
    ax.set_xlabel("prediction error (pred - true) cycles")
    ax.set_ylabel("count")
    ax.set_title("RUL prediction error distribution — 100 test engines")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/error_distribution.png", dpi=120)
    plt.close()

