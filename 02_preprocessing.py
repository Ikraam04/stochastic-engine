import numpy as np
import matplotlib.pyplot as plt
import os

model     = "linear"
run_id    = "E10_S11"
engine_id = 10
out_dir   = f"results/{model}/{run_id}/figures"
samp_dir  = f"results/{model}/{run_id}/samples"
os.makedirs(out_dir,  exist_ok=True)
os.makedirs(samp_dir, exist_ok=True)

# col layout: unit(0), cycle(1), os1-3(2-4), s1(5)...s11(15)...s21(25)
train_data = np.loadtxt("CMAPSSData/train_FD001.txt")
test_data  = np.loadtxt("CMAPSSData/test_FD001.txt")

# normalisation is fit on training data only, then applied to test
# using global train min/max means the failure threshold (s11 near max) means the same thing
# on both train and test. per-engine normalisation on test would break this — you'd have
# no reference point for where failure actually is since test trajectories are truncated
s11_col       = 15
s11_min       = train_data[:, s11_col].min()
s11_max       = train_data[:, s11_col].max()

test_engine = test_data[test_data[:, 0] == engine_id]
cycles_raw  = test_engine[:, 1]
s11_raw     = test_engine[:, s11_col]

# normalise cycles by global max — keeps beta on a consistent scale across engines
# whether an engine has 50 cycles or 300, t is in [0, 1] so the prior on beta transfers
global_max_cycle = train_data[:, 1].max()
cycles = cycles_raw / global_max_cycle

# normalise s11: 0 = healthiest seen in training, 1 = failure level
s11_norm = (s11_raw - s11_min) / (s11_max - s11_min)

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
plt.savefig(f"{out_dir}/preprocessing_s11_engine{engine_id}.png", dpi=120)
plt.close()

# save clean arrays for the sampler
np.save(f"{samp_dir}/cycles_engine{engine_id}.npy", cycles)
np.save(f"{samp_dir}/y_obs_engine{engine_id}.npy",  s11_norm)

# denormalisation info needed in 05 to convert normalised RUL back to real cycles
# rul_real = (t_failure - t_last) * global_max_cycle
cycle_last_raw = cycles_raw.max()
np.save(f"{samp_dir}/cycle_last_raw_engine{engine_id}.npy", np.array([cycle_last_raw]))
np.save(f"{samp_dir}/global_max_cycle.npy", np.array([global_max_cycle]))
