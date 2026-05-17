import numpy as np
import matplotlib.pyplot as plt
import os

model     = "exponential"
run_id    = "E10_S11"
engine_id = 10
fig_dir  = f"results/{model}/{run_id}/figures"
samp_dir = f"results/{model}/{run_id}/samples"
os.makedirs(fig_dir, exist_ok=True)

# samples stored as (alpha, phi, psi) in log space
# phi = log(beta), psi = log(gamma)
samples = np.load(f"{samp_dir}/hmc_samples_engine{engine_id}.npy")
delta_H = np.load(f"{samp_dir}/delta_H_engine{engine_id}.npy")

burn_in = 1000
post    = samples[burn_in:]
alpha_s = post[:, 0]
phi_s   = post[:, 1]
psi_s   = post[:, 2]

# recover original params for interpretable plots
beta_s  = np.exp(phi_s)
gamma_s = np.exp(psi_s)

print(f"total samples: {len(samples)}  |  post burn-in: {len(post)}")

# trace plot — 3 params now, looking for fuzzy caterpillar in all three
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].plot(alpha_s, linewidth=0.5, color="steelblue")
axes[0].axhline(alpha_s.mean(), color="red", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("alpha")
axes[0].set_title(f"trace - alpha  (mean={alpha_s.mean():.4f}, std={alpha_s.std():.4f})")

axes[1].plot(beta_s, linewidth=0.5, color="tomato")
axes[1].axhline(beta_s.mean(), color="red", linewidth=0.8, linestyle="--")
axes[1].set_ylabel("beta")
axes[1].set_title(f"trace - beta  (mean={beta_s.mean():.4f}, std={beta_s.std():.4f})")

axes[2].plot(gamma_s, linewidth=0.5, color="seagreen")
axes[2].axhline(gamma_s.mean(), color="red", linewidth=0.8, linestyle="--")
axes[2].set_ylabel("gamma")
axes[2].set_xlabel("iteration (post burn-in)")
axes[2].set_title(f"trace - gamma  (mean={gamma_s.mean():.4f}, std={gamma_s.std():.4f})")

plt.tight_layout()
plt.savefig(f"{fig_dir}/trace_plot.png", dpi=120)
plt.close()
print("saved trace_plot.png")

# delta H — same as linear, should be centred near 0
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(delta_H, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="red", linewidth=1, linestyle="--", label="zero")
ax.axvline(delta_H.mean(), color="orange", linewidth=1, linestyle="--",
           label=f"mean={delta_H.mean():.3f}")
ax.set_xlabel("delta H  (H_old - H_new)")
ax.set_ylabel("count")
ax.set_title("energy conservation - should be centred near 0")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/delta_H.png", dpi=120)
plt.close()
print("saved delta_H.png")

# ESS — effective sample size
# ESS = N / (1 + 2 * sum of autocorrelations) — how many independent samples we effectively have
# want ESS > 400
def compute_ess(chain):
    n = len(chain)
    c = chain - chain.mean()
    acf = np.correlate(c, c, mode="full")
    acf = acf[n-1:]
    acf = acf / acf[0]
    acf_sum = 0
    for k in range(1, n):
        if acf[k] < 0:
            break
        acf_sum += acf[k]
    return n / (1 + 2 * acf_sum)

ess_alpha = compute_ess(alpha_s)
ess_beta  = compute_ess(beta_s)
ess_gamma = compute_ess(gamma_s)

print(f"\nESS alpha: {ess_alpha:.0f}  {'ok' if ess_alpha > 400 else '!! too low'}")
print(f"ESS beta:  {ess_beta:.0f}  {'ok' if ess_beta  > 400 else '!! too low'}")
print(f"ESS gamma: {ess_gamma:.0f}  {'ok' if ess_gamma > 400 else '!! too low'}")
print(f"(want > 400 for reliable posterior estimates)")

# autocorrelation plots — 3 params
max_lag = 100

def get_acf(chain, max_lag):
    n = len(chain)
    c = chain - chain.mean()
    acf_full = np.correlate(c, c, mode="full")[n-1:]
    acf_full = acf_full / acf_full[0]
    return acf_full[:max_lag]

acf_alpha = get_acf(alpha_s, max_lag)
acf_beta  = get_acf(beta_s,  max_lag)
acf_gamma = get_acf(gamma_s, max_lag)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(range(max_lag), acf_alpha, color="steelblue", width=0.8)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title(f"autocorrelation - alpha  (ESS={ess_alpha:.0f})")
axes[0].set_xlabel("lag"); axes[0].set_ylabel("correlation")
axes[0].set_ylim(-0.2, 1.0)

axes[1].bar(range(max_lag), acf_beta, color="tomato", width=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title(f"autocorrelation - beta  (ESS={ess_beta:.0f})")
axes[1].set_xlabel("lag")
axes[1].set_ylim(-0.2, 1.0)

axes[2].bar(range(max_lag), acf_gamma, color="seagreen", width=0.8)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_title(f"autocorrelation - gamma  (ESS={ess_gamma:.0f})")
axes[2].set_xlabel("lag")
axes[2].set_ylim(-0.2, 1.0)

plt.tight_layout()
plt.savefig(f"{fig_dir}/autocorrelation.png", dpi=120)
plt.close()
print("saved autocorrelation.png")

# posterior histograms — show recovered beta and gamma (not log-space)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, vals, name, color in zip(
    axes,
    [alpha_s, beta_s, gamma_s],
    ["alpha", "beta", "gamma"],
    ["steelblue", "tomato", "seagreen"]
):
    ax.hist(vals, bins=60, color=color, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(vals.mean(), color="red",    linewidth=1.2, linestyle="--",
               label=f"mean={vals.mean():.4f}")
    ax.axvline(np.percentile(vals, 5),  color="orange", linewidth=1, linestyle=":",
               label="5/95%")
    ax.axvline(np.percentile(vals, 95), color="orange", linewidth=1, linestyle=":")
    ax.set_title(f"posterior - {name}")
    ax.set_xlabel(name)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{fig_dir}/posterior_histograms.png", dpi=120)
plt.close()
print("saved posterior_histograms.png")

# posterior summary
print(f"\nposterior summary (post burn-in):")
print(f"  alpha: mean={alpha_s.mean():.4f}  std={alpha_s.std():.4f}  "
      f"90% CI=[{np.percentile(alpha_s,5):.4f}, {np.percentile(alpha_s,95):.4f}]")
print(f"  beta:  mean={beta_s.mean():.4f}  std={beta_s.std():.4f}  "
      f"90% CI=[{np.percentile(beta_s,5):.4f}, {np.percentile(beta_s,95):.4f}]")
print(f"  gamma: mean={gamma_s.mean():.4f}  std={gamma_s.std():.4f}  "
      f"90% CI=[{np.percentile(gamma_s,5):.4f}, {np.percentile(gamma_s,95):.4f}]")
