import numpy as np
import matplotlib.pyplot as plt
import os

data_path = "CMAPSSData/train_FD001.txt"
out_dir = "results/figures"
os.makedirs(out_dir, exist_ok=True)

# col layout: unit, cycle, os1, os2, os3, s1...s21
data = np.loadtxt(data_path)

# quick sanity check
n_engines = int(data[:, 0].max())
print(f"rows: {len(data)}")
print(f"engines: {n_engines}")
print(f"cols: {data.shape[1]}")

# grab engine 1 to plot all its sensors
engine1 = data[data[:, 0] == 1]
cycles = engine1[:, 1]
print(f"engine 1 has {len(engine1)} cycles")

# sensor cols are 5 to 25 (s1 to s21)
sensor_cols = list(range(5, 26))

# plot all 21 sensors for engine 1 so we can see which ones drift
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
out = f"{out_dir}/sensor_overview_engine1.png"
plt.savefig(out, dpi=120)
plt.close()
print(f"saved {out}")

# calc variance for each sensor across ALL engines
# low variance = flat sensor = useless for us
print("\nsensor variances:")
for i, col in enumerate(sensor_cols):
    var = np.var(data[:, col])
    flat = " << flat, skip this one" if var < 0.01 else ""
    print(f"  s{i+1}: {var:.4f}{flat}")

# correlation with cycle - this is the real filter
# we want sensors that actually move WITH time (degrade as cycles increase)
# pearson corr close to +1 or -1 = sensor tracks degradation. near 0 = random noise
print("\nsensor correlation with cycle number (across all engines):")
all_cycles = data[:, 1]  # cycle col for every row

corrs = []
for i, col in enumerate(sensor_cols):
    sensor_vals = data[:, col]
    # np.corrcoef returns a 2x2 matrix, [0,1] is the cross-correlation
    corr = np.corrcoef(all_cycles, sensor_vals)[0, 1]
    corrs.append((abs(corr), corr, i+1))  # store abs for sorting, raw for sign

# sort by absolute correlation - highest first
corrs.sort(reverse=True)
for abs_corr, raw_corr, sensor_num in corrs:
    direction = "increases" if raw_corr > 0 else "decreases"
    print(f"  s{sensor_num}: |corr| = {abs_corr:.3f}  ({direction} with cycle)")

# pull out the non-flat ones for the overlay plot
# using variance < 0.01 as the flat cutoff same as before
signal_sensors = []
for i, col in enumerate(sensor_cols):
    if np.var(data[:, col]) >= 0.01:
        signal_sensors.append(i)  # 0-indexed within s1..s21

# inter-sensor correlation matrix for the signal sensors only
# we want sensors with high corr to cycle but LOW corr to each other (independent info)
print("\ninter-sensor correlation matrix (signal sensors only):")
sig_cols = [5 + i for i in signal_sensors]
sig_names = [f"s{i+1}" for i in signal_sensors]
sig_data = data[:, sig_cols]
inter_corr = np.corrcoef(sig_data.T)  # shape: (n_signal_sensors, n_signal_sensors)

# print it as a simple grid
header = "      " + "".join(f"{n:>7}" for n in sig_names)
print(header)
for row_i, name in enumerate(sig_names):
    row = f"{name:<6}" + "".join(f"{inter_corr[row_i, col_i]:>7.2f}" for col_i in range(len(sig_names)))
    print(row)

# overlay 10 engines on each signal sensor
# looking for sensors where ALL engines trend the same direction - thats degradation signal
sample_engines = list(range(1, 11))

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for plot_i, s_idx in enumerate(signal_sensors):
    ax = axes[plot_i]
    col = 5 + s_idx
    for eng in sample_engines:
        mask = data[:, 0] == eng
        ax.plot(data[mask, 1], data[mask, col], linewidth=0.6, alpha=0.6)
    ax.set_title(f"s{s_idx+1}")
    ax.set_xlabel("cycle")

# hide leftover empty subplots
for plot_i in range(len(signal_sensors), len(axes)):
    axes[plot_i].set_visible(False)

fig.suptitle("non-flat sensors - 10 engines overlaid") 
plt.tight_layout()
out = f"{out_dir}/signal_sensors_overlay.png"
plt.savefig(out, dpi=120)
plt.close()
print(f"saved {out}")

#based on the above, s11 is the best sensor - it has a strong correlation with cycle and decent independence from the other sensors. we'll use it as our health proxy in the HMC sampler.
