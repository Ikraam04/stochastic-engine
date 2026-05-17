import numpy as np
import matplotlib.pyplot as plt
import os

model     = "exponential"
run_id    = "E10_S11"
engine_id = 10
samp_dir  = f"results/{model}/{run_id}/samples"
fig_dir   = f"results/{model}/{run_id}/figures"
os.makedirs(fig_dir, exist_ok=True)

# samples stored as (alpha, phi, psi) — phi=log(beta), psi=log(gamma)
samples          = np.load(f"{samp_dir}/hmc_samples_engine{engine_id}.npy")
global_max_cycle = np.load(f"results/linear/{run_id}/samples/global_max_cycle.npy")[0]
cycle_last_raw   = np.load(f"results/linear/{run_id}/samples/cycle_last_raw_engine{engine_id}.npy")[0]

burn_in = 1000
post    = samples[burn_in:]
alpha_s = post[:, 0]
phi_s   = post[:, 1]
psi_s   = post[:, 2]

# recover beta and gamma from log-space
beta_s  = np.exp(phi_s)
gamma_s = np.exp(psi_s)

t_last = cycle_last_raw / global_max_cycle

# convert posterior samples to a RUL distribution
# model: h(t) = alpha + beta*exp(gamma*t), failure when h(t_fail) = failure_threshold
#
# invert for t_fail:
#   failure_threshold - alpha = beta * exp(gamma * t_fail)
#   (failure_threshold - alpha) / beta = exp(gamma * t_fail)
#   t_fail = (1/gamma) * log((failure_threshold - alpha) / beta)
#
# RUL = (t_fail - t_last) * global_max_cycle
#
# this propagates the full posterior uncertainty in (alpha, beta, gamma) through
# to a distribution over RUL — each sample gives one RUL value, 9000 samples give
# a full distribution over when this engine will fail
failure_threshold = 0.792

# the log is only defined when (threshold - alpha) > 0
# beta > 0 is guaranteed by the reparametrisation (exp() is always positive)
numerator = failure_threshold - alpha_s
valid_log = numerator > 0

rul_real = np.full(len(alpha_s), np.nan)
rul_real[valid_log] = (
    (1 / gamma_s[valid_log]) * np.log(numerator[valid_log] / beta_s[valid_log]) - t_last
) * global_max_cycle

# drop samples where the engine is predicted to have already failed (RUL <= 0)
valid    = rul_real > 0
rul_real = rul_real[valid]

true_rul_all = np.loadtxt("CMAPSSData/RUL_FD001.txt")
true_rul     = true_rul_all[engine_id - 1]

mean_rul = rul_real.mean()
ci_low   = np.percentile(rul_real, 5)
ci_high  = np.percentile(rul_real, 95)

print(f"engine {engine_id} RUL prediction (exponential model):")
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
ax.set_title(f"engine {engine_id} RUL distribution — exponential model")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{fig_dir}/rul_distribution_engine{engine_id}.png", dpi=120)
plt.close()
