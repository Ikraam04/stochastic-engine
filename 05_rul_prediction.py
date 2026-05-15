import numpy as np
import matplotlib.pyplot as plt
import os

# change this to switch runs
run_id    = "E20_S11"
engine_id = 20
samp_dir  = f"results/{run_id}/samples"
fig_dir   = f"results/{run_id}/figures"
os.makedirs(fig_dir, exist_ok=True)

samples        = np.load(f"{samp_dir}/hmc_samples_engine{engine_id}.npy")
global_max_cycle = np.load(f"{samp_dir}/global_max_cycle.npy")[0]
cycle_last_raw = np.load(f"{samp_dir}/cycle_last_raw_engine{engine_id}.npy")[0]

burn_in  = 1000
post     = samples[burn_in:]
alpha_s  = post[:, 0]
beta_s   = post[:, 1]

t_last = cycle_last_raw / global_max_cycle   # last observed cycle in normalised global coords

print(f"posterior samples (post burn-in): {len(alpha_s)}")
print(f"cycle_last_raw={cycle_last_raw:.0f}  global_max_cycle={global_max_cycle:.0f}  t_last={t_last:.4f}")

# compute RUL for every posterior sample
# h(t) = alpha + beta*t, failure when h(t) = failure_threshold
# t_failure = (threshold - alpha) / beta  in global normalised coords
# RUL = (t_failure - t_last) * global_max_cycle
failure_threshold = 0.792  # mean s11_norm at failure across 100 training engines
rul_norm = (failure_threshold - alpha_s) / beta_s
rul_real = (rul_norm - t_last) * global_max_cycle

# filter out negative RUL (samples where beta <= 0)
valid    = rul_real > 0
print(f"valid RUL samples: {valid.sum()} / {len(rul_real)}")
rul_real = rul_real[valid]

# true RUL from the label file
true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")
true_rul     = true_rul_all[engine_id - 1]

mean_rul  = rul_real.mean()
ci_low    = np.percentile(rul_real, 5)
ci_high   = np.percentile(rul_real, 95)

print(f"\nRUL prediction for engine {engine_id}:")
print(f"  mean RUL:  {mean_rul:.1f} cycles")
print(f"  90% CI:    [{ci_low:.1f}, {ci_high:.1f}] cycles")
print(f"  true RUL:  {true_rul} cycles")
print(f"  error:     {abs(mean_rul - true_rul):.1f} cycles")

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
ax.set_title(f"engine {engine_id} RUL distribution")
ax.legend(fontsize=9)

plt.tight_layout()
out = f"{fig_dir}/rul_distribution_engine{engine_id}.png"
plt.savefig(out, dpi=120)
plt.close()
print(f"saved {out}")
