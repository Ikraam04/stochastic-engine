import numpy as np
import matplotlib.pyplot as plt
import os

model     = "exponential"
run_id    = "E10_S11"
engine_id = 10
samp_dir  = f"results/{model}/{run_id}/samples"
fig_dir   = f"results/{model}/{run_id}/figures"
os.makedirs(fig_dir, exist_ok=True)

# samples are stored as (alpha, phi, psi) in log space
# need to recover beta = exp(phi) and gamma = exp(psi) before computing RUL
samples        = np.load(f"{samp_dir}/hmc_samples_engine{engine_id}.npy")
global_max_cycle = np.load(f"results/linear/{run_id}/samples/global_max_cycle.npy")[0]
cycle_last_raw   = np.load(f"results/linear/{run_id}/samples/cycle_last_raw_engine{engine_id}.npy")[0]

burn_in = 1000
post    = samples[burn_in:]
alpha_s = post[:, 0]
phi_s   = post[:, 1]
psi_s   = post[:, 2]

# recover beta and gamma from log-space samples
beta_s  = np.exp(phi_s)
gamma_s = np.exp(psi_s)

t_last = cycle_last_raw / global_max_cycle

print(f"posterior samples (post burn-in): {len(alpha_s)}")
print(f"cycle_last_raw={cycle_last_raw:.0f}  global_max_cycle={global_max_cycle:.0f}  t_last={t_last:.4f}")
print(f"beta  — mean={beta_s.mean():.4f}  std={beta_s.std():.4f}")
print(f"gamma — mean={gamma_s.mean():.4f}  std={gamma_s.std():.4f}")

# compute RUL for every posterior sample
# h(t) = alpha + beta*exp(gamma*t), failure when h(t) = threshold
# solve for t: t_fail = (1/gamma) * log((threshold - alpha) / beta)
# RUL = (t_fail - t_last) * global_max_cycle
failure_threshold = 0.792   # mean s11_norm at failure across 100 training engines

# need (threshold - alpha) > 0 and beta > 0 for log to be valid
# beta > 0 is guaranteed by the reparametrisation
# (threshold - alpha) > 0 should hold if alpha hasn't overshot the threshold
numerator = failure_threshold - alpha_s
valid_log  = (numerator > 0) & (beta_s > 0)

rul_real = np.full(len(alpha_s), np.nan)
rul_real[valid_log] = (
    (1 / gamma_s[valid_log]) * np.log(numerator[valid_log] / beta_s[valid_log]) - t_last
) * global_max_cycle

# filter: keep only positive RUL (negative means model thinks engine already failed)
valid     = rul_real > 0
print(f"valid RUL samples: {valid.sum()} / {len(rul_real)}")
rul_real  = rul_real[valid]

true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")
true_rul     = true_rul_all[engine_id - 1]

mean_rul = rul_real.mean()
ci_low   = np.percentile(rul_real, 5)
ci_high  = np.percentile(rul_real, 95)

print(f"\nRUL prediction for engine {engine_id} (exponential model):")
print(f"  90% confident this engine fails within the next {ci_high:.0f} cycles")
print(f"  best estimate (mean): {mean_rul:.1f} cycles")
print(f"  90% CI: [{ci_low:.1f}, {ci_high:.1f}] cycles")
print(f"  true RUL: {true_rul} cycles")
print(f"  error: {abs(mean_rul - true_rul):.1f} cycles")

# plot RUL distribution
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(rul_real, bins=80, density=True, color="steelblue",
        edgecolor="white", linewidth=0.3, label="RUL distribution")
ax.axvline(mean_rul, color="red",    linewidth=1.5, linestyle="--",
           label=f"mean = {mean_rul:.1f}")
ax.axvline(ci_low,   color="orange", linewidth=1.2, linestyle=":",
           label=f"90% CI = [{ci_low:.1f}, {ci_high:.1f}]")
ax.axvline(ci_high,  color="orange", linewidth=1.2, linestyle=":")
ax.axvline(true_rul, color="green",  linewidth=1.5, linestyle="-",
           label=f"true RUL = {true_rul}")

ax.set_xlabel("remaining useful life (cycles)")
ax.set_ylabel("density")
ax.set_title(f"engine {engine_id} RUL distribution — exponential model")
ax.legend(fontsize=9)

plt.tight_layout()
out = f"{fig_dir}/rul_distribution_engine{engine_id}.png"
plt.savefig(out, dpi=120)
plt.close()
print(f"saved {out}")
