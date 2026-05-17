import numpy as np
import matplotlib.pyplot as plt
import os

model     = "linear"
run_id    = "E10_S11"
engine_id = 10
fig_dir   = f"results/{model}/{run_id}/figures"
samp_dir  = f"results/{model}/{run_id}/samples"
os.makedirs(fig_dir, exist_ok=True)

samples = np.load(f"{samp_dir}/hmc_samples_engine{engine_id}.npy")
delta_H = np.load(f"{samp_dir}/delta_H_engine{engine_id}.npy")

burn_in = 1000
post    = samples[burn_in:]   # drop burn-in before checking anything
alpha_s = post[:, 0]
beta_s  = post[:, 1]

# trace plot
# plot param value at each iteration post burn-in
# what we want: a "fuzzy caterpillar" — random variation around a stable mean, no trend
# drift = chain hasn't converged yet (burn-in was too short)
# flat stretches = chain got stuck, not exploring (epsilon too big, or posterior is weird)
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

axes[0].plot(alpha_s, linewidth=0.5, color="steelblue")
axes[0].axhline(alpha_s.mean(), color="red", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("alpha")
axes[0].set_title(f"trace - alpha  (mean={alpha_s.mean():.4f}, std={alpha_s.std():.4f})")

axes[1].plot(beta_s, linewidth=0.5, color="tomato")
axes[1].axhline(beta_s.mean(), color="red", linewidth=0.8, linestyle="--")
axes[1].set_ylabel("beta")
axes[1].set_xlabel("iteration (post burn-in)")
axes[1].set_title(f"trace - beta  (mean={beta_s.mean():.4f}, std={beta_s.std():.4f})")

plt.tight_layout()
plt.savefig(f"{fig_dir}/trace_plot.png", dpi=120)
plt.close()

# delta H
# H = -log P(theta|D) + p^2/2  (hamiltonian = potential + kinetic energy)
# leapfrog should conserve H almost exactly so delta_H = H_old - H_new should be near 0
# this is WHY hmc accepts so often — proposals are almost always good
# if mean is systematically negative: leapfrog is losing energy — epsilon is too big
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(delta_H, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(0,            color="red",    linewidth=1, linestyle="--", label="zero")
ax.axvline(delta_H.mean(), color="orange", linewidth=1, linestyle="--", label=f"mean={delta_H.mean():.3f}")
ax.set_xlabel("delta H  (H_old - H_new)")
ax.set_ylabel("count")
ax.set_title("energy conservation - should be centred near 0")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/delta_H.png", dpi=120)
plt.close()

# ESS (effective sample size)
# consecutive MCMC samples are correlated — the chain doesn't jump freely, it wanders
# so 9000 post-burn-in samples != 9000 independent samples
# ESS = N / (1 + 2 * sum_k rho_k)  where rho_k is autocorrelation at lag k
# tells you how many independent draws the chain is equivalent to
# we want ESS > 400 for reliable posterior estimates
def compute_ess(chain):
    n = len(chain)
    c = chain - chain.mean()
    acf = np.correlate(c, c, mode="full")
    acf = acf[n-1:]
    acf = acf / acf[0]

    # geyer's initial monotone criterion: stop summing once acf goes negative
    # summing past this point inflates ESS artificially
    acf_sum = 0
    for k in range(1, n):
        if acf[k] < 0:
            break
        acf_sum += acf[k]

    return n / (1 + 2 * acf_sum)

ess_alpha = compute_ess(alpha_s)
ess_beta  = compute_ess(beta_s)

print(f"ESS alpha: {ess_alpha:.0f}  {'ok' if ess_alpha > 400 else '!! too low'}")
print(f"ESS beta:  {ess_beta:.0f}  {'ok' if ess_beta  > 400 else '!! too low'}")

# autocorrelation plot
# rho_k = corr(sample_i, sample_{i+k}) — how correlated is the chain k steps later
# want this to drop to ~0 quickly (few lags)
# slow decay = high correlation = chain is crawling, not exploring = low ESS
max_lag = 100

def get_acf(chain, max_lag):
    n = len(chain)
    c = chain - chain.mean()
    acf_full = np.correlate(c, c, mode="full")[n-1:]
    acf_full = acf_full / acf_full[0]
    return acf_full[:max_lag]

acf_alpha = get_acf(alpha_s, max_lag)
acf_beta  = get_acf(beta_s,  max_lag)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(max_lag), acf_alpha, color="steelblue", width=0.8)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title(f"autocorrelation - alpha  (ESS={ess_alpha:.0f})")
axes[0].set_xlabel("lag")
axes[0].set_ylabel("correlation")
axes[0].set_ylim(-0.2, 1.0)

axes[1].bar(range(max_lag), acf_beta, color="tomato", width=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title(f"autocorrelation - beta  (ESS={ess_beta:.0f})")
axes[1].set_xlabel("lag")
axes[1].set_ylim(-0.2, 1.0)

plt.tight_layout()
plt.savefig(f"{fig_dir}/autocorrelation.png", dpi=120)
plt.close()

# posterior histograms
# shows the actual shape of what we sampled — the distribution over (alpha, beta)
# width = our uncertainty. narrow = data pins it down well. wide = data is consistent with many values
# orange lines are the 5th and 95th percentile = 90% credible interval for each param
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(alpha_s, bins=60, color="steelblue", edgecolor="white", linewidth=0.3, density=True)
axes[0].axvline(alpha_s.mean(), color="red",    linewidth=1.2, linestyle="--", label=f"mean={alpha_s.mean():.4f}")
axes[0].axvline(np.percentile(alpha_s,  5), color="orange", linewidth=1, linestyle=":", label="5/95%")
axes[0].axvline(np.percentile(alpha_s, 95), color="orange", linewidth=1, linestyle=":")
axes[0].set_title("posterior - alpha (engine starting health)")
axes[0].set_xlabel("alpha")
axes[0].legend(fontsize=8)

axes[1].hist(beta_s, bins=60, color="tomato", edgecolor="white", linewidth=0.3, density=True)
axes[1].axvline(beta_s.mean(), color="red",    linewidth=1.2, linestyle="--", label=f"mean={beta_s.mean():.4f}")
axes[1].axvline(np.percentile(beta_s,  5), color="orange", linewidth=1, linestyle=":", label="5/95%")
axes[1].axvline(np.percentile(beta_s, 95), color="orange", linewidth=1, linestyle=":")
axes[1].set_title("posterior - beta (degradation rate)")
axes[1].set_xlabel("beta")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{fig_dir}/posterior_histograms.png", dpi=120)
plt.close()

print(f"alpha: mean={alpha_s.mean():.4f}  std={alpha_s.std():.4f}  90% CI=[{np.percentile(alpha_s,5):.4f}, {np.percentile(alpha_s,95):.4f}]")
print(f"beta:  mean={beta_s.mean():.4f}  std={beta_s.std():.4f}  90% CI=[{np.percentile(beta_s,5):.4f}, {np.percentile(beta_s,95):.4f}]")
