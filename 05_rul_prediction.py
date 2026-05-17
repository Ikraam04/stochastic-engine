import numpy as np
import matplotlib.pyplot as plt
import os

model     = "linear"
run_id    = "E10_S11"
engine_id = 10
samp_dir  = f"results/{model}/{run_id}/samples"
fig_dir   = f"results/{model}/{run_id}/figures"
os.makedirs(fig_dir, exist_ok=True)

samples          = np.load(f"{samp_dir}/hmc_samples_engine{engine_id}.npy")
global_max_cycle = np.load(f"{samp_dir}/global_max_cycle.npy")[0]
cycle_last_raw   = np.load(f"{samp_dir}/cycle_last_raw_engine{engine_id}.npy")[0]

burn_in = 1000
post    = samples[burn_in:]
alpha_s = post[:, 0]
beta_s  = post[:, 1]

# t_last: the last observed cycle in normalised coords
# we have seen data up to this point — RUL is everything after
t_last = cycle_last_raw / global_max_cycle

# convert posterior samples to a RUL distribution
# model: h(t) = alpha + beta*t, failure when h(t_fail) = failure_threshold
# invert:
#   failure_threshold = alpha + beta * t_fail
#   t_fail = (failure_threshold - alpha) / beta   (in normalised cycle coords)
#
# RUL in normalised coords = t_fail - t_last  (time remaining after last observation)
# RUL in real cycles = RUL_norm * global_max_cycle
#
# doing this for every posterior sample (alpha_i, beta_i) maps the full posterior
# uncertainty in the params directly to uncertainty in RUL
# so the output is a DISTRIBUTION over when the engine will fail, not a point estimate
failure_threshold = 0.792   # mean s11_norm at last cycle across 100 training engines

rul_norm = (failure_threshold - alpha_s) / beta_s
rul_real = (rul_norm - t_last) * global_max_cycle

# drop samples where predicted failure is in the past (RUL <= 0)
# these happen when beta is near zero or negative — linear model can produce these
valid    = rul_real > 0
rul_real = rul_real[valid]

true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")
true_rul     = true_rul_all[engine_id - 1]

mean_rul = rul_real.mean()
ci_low   = np.percentile(rul_real, 5)
ci_high  = np.percentile(rul_real, 95)

print(f"engine {engine_id} RUL prediction:")
print(f"  90% CI: [{ci_low:.1f}, {ci_high:.1f}] cycles")
print(f"  mean:   {mean_rul:.1f} cycles")
print(f"  true:   {true_rul} cycles")
print(f"  error:  {abs(mean_rul - true_rul):.1f} cycles")

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
plt.savefig(f"{fig_dir}/rul_distribution_engine{engine_id}.png", dpi=120)
plt.close()

