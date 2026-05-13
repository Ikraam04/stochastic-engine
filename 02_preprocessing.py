import numpy as np
import matplotlib.pyplot as plt
import os

#sanity check: load the data, filter to one engine, normalise s11, and plot before/after normalisation, just for s11 on engine 1 for now.

data_path = "CMAPSSData/train_FD001.txt"
out_dir = "results/figures"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("results/samples", exist_ok=True)

# col layout: unit(0), cycle(1), os1-3(2-4), s1(5), s2(6) ... s11(15) ... s21(25)
data = np.loadtxt(data_path)

# --- filter to one engine ---
# starting with engine 1, sampler only handles one at a time for now
engine_id = 1
engine_data = data[data[:, 0] == engine_id]

cycles_raw = engine_data[:, 1]    # cycle numbers 1..N
s11_raw    = engine_data[:, 15]  # s11 is our health proxy (highest corr with cycle)

# normalise cycles to [0,1] too - otherwise grad_beta explodes in the sampler
# (it sums cycle * residual across all steps, cycles up to 192 makes gradients huge)
cycles = (cycles_raw - cycles_raw.min()) / (cycles_raw.max() - cycles_raw.min())

print(f"engine {engine_id}: {len(cycles_raw)} cycles")
print(f"s11 raw  — min: {s11_raw.min():.3f}  max: {s11_raw.max():.3f}  mean: {s11_raw.mean():.3f}")

# --- normalise s11 to [0, 1] and flip it ---
# s11 increases with degradation, but our health model needs h(t) to start near 0 and rise to 1
# so we just do standard min-max - low s11 (healthy) maps to 0, high s11 (failing) maps to 1
# RUL formula becomes: t_RUL = (1 - alpha) / beta
s11_min = s11_raw.min()
s11_max = s11_raw.max()
s11_norm = (s11_raw - s11_min) / (s11_max - s11_min)

print(f"s11 norm — min: {s11_norm.min():.3f}  max: {s11_norm.max():.3f}  mean: {s11_norm.mean():.3f}")

# --- sanity plot: raw vs normalised ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(cycles, s11_raw)
axes[0].set_title("s11 raw")
axes[0].set_xlabel("normalised cycle")
axes[0].set_ylabel("sensor value")

axes[1].plot(cycles, s11_norm)
axes[1].set_title("s11 normalised [0=healthy, 1=failure]")
axes[1].set_xlabel("normalised cycle")
axes[1].set_ylabel("sensor value")

fig.suptitle(f"engine {engine_id} - s11 before and after normalisation")
plt.tight_layout()
out = f"{out_dir}/preprocessing_s11_engine{engine_id}.png"
plt.savefig(out, dpi=120)
plt.close()
print(f"saved {out}")

# --- save clean arrays ---
# these are what 03_hmc_sampler.py will load
np.save(f"results/samples/cycles_engine{engine_id}.npy", cycles)
np.save(f"results/samples/y_obs_engine{engine_id}.npy", s11_norm)
print(f"saved cycles and y_obs arrays for engine {engine_id}")
