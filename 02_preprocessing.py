import numpy as np
import matplotlib.pyplot as plt
import os

# change these to switch runs
model     = "linear"
run_id    = "E10_S11"
engine_id = 10
out_dir   = f"results/{model}/{run_id}/figures"
samp_dir  = f"results/{model}/{run_id}/samples"
os.makedirs(out_dir,  exist_ok=True)
os.makedirs(samp_dir, exist_ok=True)

# col layout: unit(0), cycle(1), os1-3(2-4), s1(5), s2(6) ... s11(15) ... s21(25)
train_data = np.loadtxt("CMAPSSData/train_FD001.txt")
test_data  = np.loadtxt("CMAPSSData/test_FD001.txt")

# --- fit the normalisation scale on training data (all engines) ---
# we use global train min/max so the failure threshold (s11=1.0) means the same thing
# across train and test. if we used per-engine min/max on the test set we wouldnt know
# where failure actually is since the test trajectories are truncated before failure
s11_col = 15
s11_train_all = train_data[:, s11_col]
s11_min = s11_train_all.min()
s11_max = s11_train_all.max()
print(f"s11 normalisation range (from training): min={s11_min:.3f}  max={s11_max:.3f}")

# --- filter test set to one engine (truncated before failure - we predict the remaining RUL) ---
test_engine = test_data[test_data[:, 0] == engine_id]
cycles_raw  = test_engine[:, 1]
s11_raw     = test_engine[:, s11_col]
print(f"engine {engine_id} test trajectory: {len(cycles_raw)} cycles (truncated before failure)")
print(f"s11 raw — min: {s11_raw.min():.3f}  max: {s11_raw.max():.3f}")

# normalise cycles using global max across all training engines
# this keeps beta on a consistent scale across engines — short and long trajectories
# comparable, and the prior on beta means the same thing regardless of engine length
global_max_cycle = train_data[:, 1].max()
cycles = cycles_raw / global_max_cycle

# normalise s11 using the TRAINING min/max
# so 0 = healthy (global min), 1 = failure threshold (global max from training)
s11_norm = (s11_raw - s11_min) / (s11_max - s11_min)

print(f"s11 norm — min: {s11_norm.min():.3f}  max: {s11_norm.max():.3f}")
print(f"  (values < 1.0 since engine hasnt failed yet in the test trajectory)")

# sanity plot: raw vs normalised
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(cycles_raw, s11_raw)
axes[0].set_title("s11 raw (test trajectory)")
axes[0].set_xlabel("cycle")
axes[0].set_ylabel("sensor value")

axes[1].plot(cycles_raw, s11_norm)
axes[1].axhline(1.0, color="red", linewidth=0.8, linestyle="--", label="failure threshold")
axes[1].set_title("s11 normalised [0=healthy, 1=failure]")
axes[1].set_xlabel("cycle")
axes[1].set_ylabel("sensor value")
axes[1].legend(fontsize=8)

fig.suptitle(f"engine {engine_id} test trajectory - s11 (truncated before failure)")
plt.tight_layout()
out = f"{out_dir}/preprocessing_s11_engine{engine_id}.png"
plt.savefig(out, dpi=120)
plt.close()
print(f"saved {out}")

# save clean arrays for the sampler
np.save(f"{samp_dir}/cycles_engine{engine_id}.npy", cycles)
np.save(f"{samp_dir}/y_obs_engine{engine_id}.npy",  s11_norm)

# save denormalisation info for 05
# to go from normalised RUL back to real cycles: rul_real = (t_failure - t_last) * global_max_cycle
cycle_last_raw = cycles_raw.max()
np.save(f"{samp_dir}/cycle_last_raw_engine{engine_id}.npy", np.array([cycle_last_raw]))
np.save(f"{samp_dir}/global_max_cycle.npy", np.array([global_max_cycle]))

print(f"saved cycles and y_obs arrays to {samp_dir}")
print(f"cycle_last_raw={cycle_last_raw:.0f}  global_max_cycle={global_max_cycle:.0f}")
