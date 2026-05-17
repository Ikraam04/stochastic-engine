import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
from scipy.optimize import curve_fit

model   = "exponential"
fig_dir = f"results/{model}/fleet_S11/figures"
os.makedirs(fig_dir, exist_ok=True)

train_data   = np.loadtxt("CMAPSSData/train_FD001.txt")
test_data    = np.loadtxt("CMAPSSData/test_FD001.txt")
true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")

s11_col          = 15
s11_min          = train_data[:, s11_col].min()
s11_max          = train_data[:, s11_col].max()
global_max_cycle = train_data[:, 1].max()

# fit h(t) = alpha + beta*exp(gamma*t) to each training engine to get priors
# same as fit_exp_priors.py — clip outliers before taking mean/std
def exp_model(t, alpha, beta, gamma):
    return alpha + beta * np.exp(gamma * t)

print("fitting exponential curves to training engines to derive priors...")
alphas, betas, gammas = [], [], []
for eid in range(1, 101):
    e = train_data[train_data[:, 0] == eid]
    t = e[:, 1] / global_max_cycle
    y = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    try:
        popt, _ = curve_fit(exp_model, t, y, p0=[y[0], 0.1, 2.0], maxfev=10000)
        alphas.append(popt[0]); betas.append(popt[1]); gammas.append(popt[2])
    except RuntimeError:
        pass

def clipped_stats(arr):
    lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
    c = np.array(arr)[(np.array(arr) >= lo) & (np.array(arr) <= hi)]
    return c.mean(), c.std()

mu_alpha, sigma_alpha = clipped_stats(alphas)
mu_beta,  sigma_beta  = clipped_stats(betas)
mu_gamma, sigma_gamma = clipped_stats(gammas)

# reparametrised priors: phi=log(beta), psi=log(gamma)
mu_phi    = np.log(mu_beta)
sigma_phi = sigma_beta / mu_beta
mu_psi    = np.log(mu_gamma)
sigma_psi = sigma_gamma / mu_gamma

sigma_noise = 0.15

# failure threshold — mean s11_norm at last cycle across all training engines
end_s11 = []
for eid in range(1, 101):
    e = train_data[train_data[:, 0] == eid]
    s11_norm = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    end_s11.append(s11_norm[-1])
failure_threshold = np.mean(end_s11)

# hmc hyperparams — tuned on training engine 10, 78% acceptance
epsilon   = 0.015
L         = 15
n_samples = 10000
burn_in   = 1000


def process_engine(args):
    engine_id, cycles, y_obs, t_last, true_rul, cfg = args

    mu_a    = cfg["mu_alpha"];    sig_a   = cfg["sigma_alpha"]
    mu_ph   = cfg["mu_phi"];      sig_ph  = cfg["sigma_phi"]
    mu_ps   = cfg["mu_psi"];      sig_ps  = cfg["sigma_psi"]
    sig_n   = cfg["sigma_noise"]
    eps     = cfg["epsilon"];     Lsteps  = cfg["L"]
    ns      = cfg["n_samples"];   bi      = cfg["burn_in"]
    thresh  = cfg["failure_threshold"]
    gmc     = cfg["global_max_cycle"]

    def log_posterior(alpha, phi, psi):
        beta  = np.exp(phi)
        gamma = np.exp(psi)
        h  = alpha + beta * np.exp(np.clip(gamma * cycles, -500, 500))
        ll = -0.5 * np.sum(((y_obs - h) / sig_n) ** 2)
        lp = (-0.5 * ((alpha - mu_a)  / sig_a)  ** 2
            - 0.5 * ((phi   - mu_ph) / sig_ph) ** 2
            - 0.5 * ((psi   - mu_ps) / sig_ps) ** 2)
        return ll + lp

    def compute_gradients(alpha, phi, psi):
        beta   = np.exp(phi)
        gamma  = np.exp(psi)
        exp_gt = np.exp(np.clip(gamma * cycles, -500, 500))
        r      = y_obs - (alpha + beta * exp_gt)
        g_alpha = (1/sig_n**2)*np.sum(r)                         - (alpha-mu_a) /sig_a**2
        g_phi   = (1/sig_n**2)*np.sum(r*exp_gt)*beta             - (phi  -mu_ph)/sig_ph**2
        g_psi   = (1/sig_n**2)*np.sum(r*beta*cycles*exp_gt)*gamma- (psi  -mu_ps)/sig_ps**2
        return np.array([g_alpha, g_phi, g_psi])

    def leapfrog(theta, p):
        theta = theta.copy(); p = p.copy()
        p += (eps/2) * compute_gradients(theta[0], theta[1], theta[2])
        for _ in range(Lsteps - 1):
            theta += eps * p
            p     += eps * compute_gradients(theta[0], theta[1], theta[2])
        theta += eps * p
        p     += (eps/2) * compute_gradients(theta[0], theta[1], theta[2])
        return theta, p

    theta    = np.array([mu_a, mu_ph, mu_ps])
    samples  = np.zeros((ns, 3))
    accepted = 0

    for i in range(ns):
        p      = np.random.randn(3)
        H_cur  = -log_posterior(*theta) + 0.5 * np.dot(p, p)
        th_p, p_p = leapfrog(theta, p)
        H_prop = -log_posterior(*th_p)  + 0.5 * np.dot(p_p, p_p)
        if np.log(np.random.rand()) < H_cur - H_prop:
            theta = th_p; accepted += 1
        samples[i] = theta

    acc     = accepted / ns
    post    = samples[bi:]
    alpha_s = post[:, 0]
    beta_s  = np.exp(post[:, 1])
    gamma_s = np.exp(post[:, 2])

    # RUL: t_fail = (1/gamma) * log((threshold - alpha) / beta)
    # only valid when (threshold - alpha) > 0 and beta > 0
    numerator = thresh - alpha_s
    valid_log = numerator > 0   # beta > 0 guaranteed by reparametrisation
    rul_real  = np.full(len(alpha_s), np.nan)
    rul_real[valid_log] = (
        (1 / gamma_s[valid_log]) * np.log(numerator[valid_log] / beta_s[valid_log]) - t_last
    ) * gmc
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
    print(f"derived priors — mu_alpha={mu_alpha:.3f}  sigma_alpha={sigma_alpha:.3f}  "
          f"mu_beta={mu_beta:.4f}  sigma_beta={sigma_beta:.4f}  "
          f"mu_gamma={mu_gamma:.3f}  sigma_gamma={sigma_gamma:.3f}")
    print(f"failure threshold: {failure_threshold:.4f}")
    print(f"hyperparams — epsilon={epsilon}  L={L}  n_samples={n_samples}  burn_in={burn_in}")
    print("CI shown as 5th<=x<=95th percentile  |  target acc 60-80%")

    cfg = dict(
        mu_alpha=mu_alpha,       sigma_alpha=sigma_alpha,
        mu_phi=mu_phi,           sigma_phi=sigma_phi,
        mu_psi=mu_psi,           sigma_psi=sigma_psi,
        sigma_noise=sigma_noise, epsilon=epsilon,
        L=L, n_samples=n_samples, burn_in=burn_in,
        failure_threshold=failure_threshold,
        global_max_cycle=global_max_cycle,
    )

    engine_args = []
    for eid in range(1, 101):
        e          = test_data[test_data[:, 0] == eid]
        cycles_raw = e[:, 1]
        cycles     = cycles_raw / global_max_cycle
        y_obs      = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
        t_last     = cycles_raw.max() / global_max_cycle
        true_rul   = true_rul_all[eid - 1]
        engine_args.append((eid, cycles, y_obs, t_last, true_rul, cfg))

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
    ci_highs   = np.array([r["ci_high"]  for r in results])

    print(f"\nsummary:")
    print(f"mae:        {abs_errors.mean():.1f} cycles")
    print(f"rmse:       {np.sqrt((errors**2).mean()):.1f} cycles")
    print(f"mean error (bias): {errors.mean():+.1f} cycles")
    print(f"90% confident fleet fails within median {np.median(ci_highs):.0f} cycles  "
          f"(vs median true RUL: {np.median(true_ruls):.0f})")

    # plot 1: predicted vs true
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_ruls, pred_ruls, alpha=0.6, s=30, color="steelblue")
    lims = [0, max(true_ruls.max(), pred_ruls.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("true RUL (cycles)")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_title("predicted vs true RUL — all 100 test engines, exponential model")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/pred_vs_true_rul.png", dpi=120)
    plt.close()
    print("saved pred_vs_true_rul.png")

    # plot 2: error distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0,             color="red",    linewidth=1.2, linestyle="--", label="zero error")
    ax.axvline(errors.mean(), color="orange", linewidth=1.2, linestyle="--",
               label=f"mean={errors.mean():+.1f}")
    ax.set_xlabel("prediction error (pred - true) cycles")
    ax.set_ylabel("count")
    ax.set_title("RUL prediction error distribution — 100 test engines, exponential model")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/error_distribution.png", dpi=120)
    plt.close()
    print("saved error_distribution.png")
