import numpy as np
import matplotlib.pyplot as plt
import os

data_path = "CMAPSSData/train_FD001.txt"

run_id  = "E1_S11"
out_dir = f"results/{run_id}/figures"
os.makedirs(out_dir, exist_ok=True)

# col layout: unit, cycle, os1, os2, os3, s1...s21
data = np.loadtxt(data_path)

n_engines = int(data[:, 0].max())

# grab engine 1 to plot all sensors — just a visual check to see which ones drift
engine1 = data[data[:, 0] == 1]
cycles  = engine1[:, 1]

sensor_cols = list(range(5, 26))  # s1 to s21

fig, axes = plt.subplots(7, 3, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    ax = axes[i]
    ax.plot(cycles, engine1[:, col], linewidth=0.8)
    ax.set_title(f"s{i+1}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("value")

fig.suptitle("engine 1 - all 21 sensors over its whole life")
plt.tight_layout()
plt.savefig(f"{out_dir}/sensor_overview_engine1.png", dpi=120)
plt.close()

# variance filter — sensors with near-zero variance are completely flat across all engines
# a flat sensor gives the sampler no info about degradation so we skip them
print("sensor variances (flat ones skipped):")
for i, col in enumerate(sensor_cols):
    var = np.var(data[:, col])
    flat = " << flat" if var < 0.01 else ""
    print(f"  s{i+1}: {var:.4f}{flat}")

# correlation with cycle number
# this is the main filter for picking a health proxy
# we want a sensor that consistently increases or decreases as the engine ages
# |corr| near 1 = sensor reliably tracks degradation
# |corr| near 0 = sensor is just noise with respect to time, useless for RUL
print("\ncorrelation with cycle (we want high |corr|):")
all_cycles = data[:, 1]

corrs = []
for i, col in enumerate(sensor_cols):
    sensor_vals = data[:, col]
    # np.corrcoef returns a 2x2 matrix, [0,1] is the cross-correlation
    corr = np.corrcoef(all_cycles, sensor_vals)[0, 1]
    corrs.append((abs(corr), corr, i+1))

corrs.sort(reverse=True)
for abs_corr, raw_corr, sensor_num in corrs:
    direction = "increases" if raw_corr > 0 else "decreases"
    print(f"  s{sensor_num}: |corr| = {abs_corr:.3f}  ({direction} with cycle)")

# pull non-flat sensors for the multi-engine overlay
signal_sensors = []
for i, col in enumerate(sensor_cols):
    if np.var(data[:, col]) >= 0.01:
        signal_sensors.append(i)

# inter-sensor correlation for non-flat sensors
# checking for redundancy — if two sensors are highly correlated with each other
# they're giving the same information, no reason to use both
print("\ninter-sensor correlation (signal sensors only):")
sig_cols  = [5 + i for i in signal_sensors]
sig_names = [f"s{i+1}" for i in signal_sensors]
sig_data  = data[:, sig_cols]
inter_corr = np.corrcoef(sig_data.T)

header = "      " + "".join(f"{n:>7}" for n in sig_names)
print(header)
for row_i, name in enumerate(sig_names):
    row = f"{name:<6}" + "".join(f"{inter_corr[row_i, col_i]:>7.2f}" for col_i in range(len(sig_names)))
    print(row)

# overlay 10 engines — we want sensors where all engines trend in the same direction
# consistent direction across engines = it's tracking real degradation, not just noise
sample_engines = list(range(1, 11))

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for plot_i, s_idx in enumerate(signal_sensors):
    ax  = axes[plot_i]
    col = 5 + s_idx
    for eng in sample_engines:
        mask = data[:, 0] == eng
        ax.plot(data[mask, 1], data[mask, col], linewidth=0.6, alpha=0.6)
    ax.set_title(f"s{s_idx+1}")
    ax.set_xlabel("cycle")

for plot_i in range(len(signal_sensors), len(axes)):
    axes[plot_i].set_visible(False)

fig.suptitle("non-flat sensors - 10 engines overlaid")
plt.tight_layout()
plt.savefig(f"{out_dir}/signal_sensors_overlay.png", dpi=120)
plt.close()

# s11 is the winner: |corr|=0.634 (highest in the fleet), increases monotonically with cycle,
# consistent direction across all 10 overlaid engines, and reasonably independent from the others.
# we treat it as the noisy observation of true engine health h(t) in the HMC model.
