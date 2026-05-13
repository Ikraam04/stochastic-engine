import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results/figures", exist_ok=True)

samples   = np.load("results/samples/hmc_samples_engine1.npy")
delta_H   = np.load("results/samples/delta_H_engine1.npy")

burn_in   = 500
post      = samples[burn_in:]   # discard burn-in before checking anything
alpha_s   = post[:, 0]
beta_s    = post[:, 1]

print(f"total samples: {len(samples)}  |  post burn-in: {len(post)}")

# trace plot
# looking for a fuzzy caterpillar - no drift, no flat plateaus
# drift = chain hasn't converged. flat bits = chain got stuck
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
plt.savefig("results/figures/trace_plot.png", dpi=120)
plt.close()
print("saved trace_plot.png")

# DeltaH 
# H_old - H_new should be near 0 - leapfrog conserves energy almost perfectly
# systematically negative = leapfrog losing energy = epsilon too big
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(delta_H, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="red", linewidth=1, linestyle="--", label="zero")
ax.axvline(delta_H.mean(), color="orange", linewidth=1, linestyle="--", label=f"mean={delta_H.mean():.3f}")
ax.set_xlabel("delta H  (H_old - H_new)")
ax.set_ylabel("count")
ax.set_title("energy conservation - should be centred near 0")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/delta_H.png", dpi=120)
plt.close()
print("saved delta_H.png")

# ESS - effective sample size - how many independent samples do we have after accounting for autocorrelation in the chain?
# autocorrelation in the chain means consecutive samples are not independent
# ESS = N / (1 + 2 * sum of autocorrelations) - how many independent samples we effectively have
# want ESS > 400

def compute_ess(chain):
    n = len(chain)
    chain_centred = chain - chain.mean()
    # normalised autocorrelation at lag k
    acf = np.correlate(chain_centred, chain_centred, mode="full")
    acf = acf[n-1:]          # keep only lag >= 0
    acf = acf / acf[0]       # normalise so lag-0 = 1

    # sum autocorrelations until they go negative (geyer's initial monotone criterion)
    # summing past negative values inflates ESS artificially
    acf_sum = 0
    for k in range(1, n):
        if acf[k] < 0:
            break
        acf_sum += acf[k]

    ess = n / (1 + 2 * acf_sum)
    return ess

ess_alpha = compute_ess(alpha_s)
ess_beta  = compute_ess(beta_s)

print(f"\nESS alpha: {ess_alpha:.0f}  {'ok' if ess_alpha > 400 else '!! too low'}")
print(f"ESS beta:  {ess_beta:.0f}  {'ok' if ess_beta  > 400 else '!! too low'}")
print(f"(want > 400 for reliable posterior estimates)")

# --- quick posterior summary ---
print(f"\nposterior summary (post burn-in):")
print(f"  alpha: mean={alpha_s.mean():.4f}  std={alpha_s.std():.4f}  90% CI=[{np.percentile(alpha_s,5):.4f}, {np.percentile(alpha_s,95):.4f}]")
print(f"  beta:  mean={beta_s.mean():.4f}  std={beta_s.std():.4f}  90% CI=[{np.percentile(beta_s,5):.4f}, {np.percentile(beta_s,95):.4f}]")
