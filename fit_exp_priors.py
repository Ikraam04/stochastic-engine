import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# same normalisation as the rest of the pipeline
train_data       = np.loadtxt("CMAPSSData/train_FD001.txt")
s11_col          = 15
s11_min          = train_data[:, s11_col].min()
s11_max          = train_data[:, s11_col].max()
global_max_cycle = train_data[:, 1].max()


def exp_model(t, alpha, beta, gamma):
    # h(t) = alpha + beta * exp(gamma * t)
    return alpha + beta * np.exp(gamma * t)


alphas, betas, gammas = [], [], []
failed = []

for eid in range(1, 101):
    e = train_data[train_data[:, 0] == eid]
    t = e[:, 1] / global_max_cycle
    y = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    try:
        p0 = [y[0], 0.1, 2.0]
        popt, _ = curve_fit(exp_model, t, y, p0=p0, maxfev=10000)
        alphas.append(popt[0])
        betas.append(popt[1])
        gammas.append(popt[2])
    except RuntimeError:
        failed.append(eid)

print(f"fitted {len(alphas)} / 100 engines  ({len(failed)} failed to converge)")
if failed:
    print(f"failed engines: {failed}")

alphas = np.array(alphas)
betas  = np.array(betas)
gammas = np.array(gammas)

# clip extreme outliers before computing priors
# a few engines dont converge cleanly and would blow up the mean/std otherwise
# 5th/95th percentile clip is a simple robust estimate
def robust_stats(arr, label):
    lo, hi   = np.percentile(arr, 5), np.percentile(arr, 95)
    clipped  = arr[(arr >= lo) & (arr <= hi)]
    print(f"{label}: mean={clipped.mean():.4f}  std={clipped.std():.4f}  "
          f"(clipped {len(arr)-len(clipped)} outliers, raw range [{arr.min():.3f}, {arr.max():.3f}])")
    return clipped.mean(), clipped.std()

mu_alpha, sigma_alpha = robust_stats(alphas, "alpha")
mu_beta,  sigma_beta  = robust_stats(betas,  "beta ")
mu_gamma, sigma_gamma = robust_stats(gammas, "gamma")

# plot 1: distribution of fitted params across the fleet
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, vals, name in zip(axes, [alphas, betas, gammas], ["alpha", "beta", "gamma"]):
    ax.hist(vals, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(vals.mean(), color="red", linewidth=1.2, linestyle="--", label=f"mean={vals.mean():.3f}")
    ax.set_title(f"fitted {name} — 100 training engines")
    ax.set_xlabel(name)
    ax.legend(fontsize=8)
plt.tight_layout()
os.makedirs("results/exponential", exist_ok=True)
plt.savefig("results/exponential/exp_prior_distributions.png", dpi=120)
plt.close()

# plot 2: a few example fits to sanity check the curve shape
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
sample_ids = [1, 5, 10, 20, 30, 50, 70, 90]
for ax, eid in zip(axes.flat, sample_ids):
    e   = train_data[train_data[:, 0] == eid]
    t   = e[:, 1] / global_max_cycle
    y   = (e[:, s11_col] - s11_min) / (s11_max - s11_min)
    idx = eid - 1
    if idx < len(alphas):
        t_fit = np.linspace(t.min(), t.max(), 200)
        y_fit = exp_model(t_fit, alphas[idx], betas[idx], gammas[idx])
        ax.plot(t, y, "o", markersize=2, color="steelblue", alpha=0.6, label="data")
        ax.plot(t_fit, y_fit, color="red", linewidth=1.2, label="fit")
        ax.set_title(f"engine {eid}  gamma={gammas[idx]:.2f}")
    else:
        ax.set_title(f"engine {eid}  (failed)")
    ax.set_xlabel("t (normalised)")
    ax.set_ylabel("s11 norm")
    ax.legend(fontsize=7)
plt.suptitle("exponential fits on training engines — sanity check", fontsize=11)
plt.tight_layout()
plt.savefig("results/exponential/exp_fit_examples.png", dpi=120)
plt.close()
