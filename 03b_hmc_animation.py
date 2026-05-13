import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# change this to switch runs - must match preprocessing
run_id   = "E1_S11"
samp_dir = f"results/{run_id}/samples"
fig_dir  = f"results/{run_id}/figures"

cycles = np.load(f"{samp_dir}/cycles_engine1.npy")
y_obs  = np.load(f"{samp_dir}/y_obs_engine1.npy")

# same priors and settings as the sampler
mu_alpha, sigma_alpha = 0.1, 0.2
mu_beta,  sigma_beta  = 0.6, 0.3
sigma_noise = 0.15
epsilon     = 0.014
L           = 20
n_animate   = 300   # how many hmc steps to animate (keep it short or it's slow)


def log_posterior(alpha, beta):
    ll = -0.5 * np.sum(((y_obs - (alpha + beta * cycles)) / sigma_noise) ** 2)
    lp = -0.5 * ((alpha - mu_alpha) / sigma_alpha)**2 \
       - 0.5 * ((beta  - mu_beta)  / sigma_beta) **2
    return ll + lp

def compute_gradients(alpha, beta):
    r = y_obs - (alpha + beta * cycles)
    ga = (1/sigma_noise**2) * np.sum(r)         - (alpha - mu_alpha) / sigma_alpha**2
    gb = (1/sigma_noise**2) * np.sum(cycles * r) - (beta  - mu_beta)  / sigma_beta**2
    return np.array([ga, gb])

def leapfrog(theta, p):
    theta = theta.copy(); p = p.copy()
    p += (epsilon/2) * compute_gradients(theta[0], theta[1])
    for _ in range(L - 1):
        theta += epsilon * p
        p     += epsilon * compute_gradients(theta[0], theta[1])
    theta += epsilon * p
    p     += (epsilon/2) * compute_gradients(theta[0], theta[1])
    return theta, p


#  pre-run the sampler and store the leapfrog paths too
# we store each full leapfrog trajectory so we can animate the "ball rolling"
np.random.seed(7)
theta = np.array([mu_alpha, mu_beta])

chain         = []   # accepted positions over time
leap_paths    = []   # full leapfrog trajectory for each step (for animation)
accepted_list = []   # was each proposal accepted?

for _ in range(n_animate):
    p = np.random.randn(2)
    H_cur = -log_posterior(theta[0], theta[1]) + 0.5 * np.dot(p, p)

    # store the leapfrog path for this proposal
    path = [theta.copy()]
    th_tmp = theta.copy(); p_tmp = p.copy()
    p_tmp += (epsilon/2) * compute_gradients(th_tmp[0], th_tmp[1])
    for _ in range(L - 1):
        th_tmp += epsilon * p_tmp
        p_tmp  += epsilon * compute_gradients(th_tmp[0], th_tmp[1])
        path.append(th_tmp.copy())
    th_tmp += epsilon * p_tmp
    p_tmp  += (epsilon/2) * compute_gradients(th_tmp[0], th_tmp[1])
    path.append(th_tmp.copy())
    leap_paths.append(np.array(path))

    H_prop = -log_posterior(th_tmp[0], th_tmp[1]) + 0.5 * np.dot(p_tmp, p_tmp)
    accept = np.log(np.random.rand()) < H_cur - H_prop
    if accept:
        theta = th_tmp
    accepted_list.append(accept)
    chain.append(theta.copy())

chain = np.array(chain)

# build the contour grid for the posterior landscape
# plot the region the chain actually explored plus a bit of padding
a_lo, a_hi = chain[:, 0].min() - 0.05, chain[:, 0].max() + 0.05
b_lo, b_hi = chain[:, 1].min() - 0.1,  chain[:, 1].max() + 0.1

a_grid = np.linspace(a_lo, a_hi, 80)
b_grid = np.linspace(b_lo, b_hi, 80)
AA, BB = np.meshgrid(a_grid, b_grid)
ZZ = np.array([[log_posterior(a, b) for a in a_grid] for b in b_grid])


# animation 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("hmc sampler in action E1: S11, normalised")

ax_contour = axes[0]
ax_trace   = axes[1]

# left: posterior contour + chain path
ax_contour.contourf(AA, BB, ZZ, levels=25, cmap="Blues")
ax_contour.contour(AA,  BB, ZZ, levels=25, colors="white", linewidths=0.3, alpha=0.4)
ax_contour.set_xlabel("alpha")
ax_contour.set_ylabel("beta")
ax_contour.set_title("posterior landscape + leapfrog paths")

# right: trace plots
ax_trace.set_xlabel("iteration")
ax_trace.set_title("chain trace")
ax_trace.set_xlim(0, n_animate)
ax_trace.set_ylim(chain[:, 0].min() - 0.05, chain[:, 1].max() + 0.1)
ax_trace.axhline(chain[:, 0].mean(), color="steelblue", linestyle="--", linewidth=0.8, alpha=0.5)
ax_trace.axhline(chain[:, 1].mean(), color="tomato",    linestyle="--", linewidth=0.8, alpha=0.5)

# plot elements we'll update each frame
chain_line,    = ax_contour.plot([], [], "w-",  linewidth=0.6, alpha=0.5)
chain_dot,     = ax_contour.plot([], [], "wo",  markersize=4)
leap_line,     = ax_contour.plot([], [], color="yellow", linewidth=1.2, alpha=0.8)
leap_dot,      = ax_contour.plot([], [], "y*",  markersize=8)

alpha_line,    = ax_trace.plot([], [], color="steelblue", linewidth=0.8, label="alpha")
beta_line,     = ax_trace.plot([], [], color="tomato",    linewidth=0.8, label="beta")
ax_trace.legend(loc="upper left", fontsize=8)

status_text = ax_contour.text(0.02, 0.97, "", transform=ax_contour.transAxes,
                               color="white", fontsize=8, va="top")


def update(frame):
    # draw chain path up to this frame
    chain_line.set_data(chain[:frame, 0], chain[:frame, 1])
    chain_dot.set_data([chain[frame, 0]], [chain[frame, 1]])

    # draw leapfrog trajectory for this proposal
    path = leap_paths[frame]
    leap_line.set_data(path[:, 0], path[:, 1])
    leap_dot.set_data([path[-1, 0]], [path[-1, 1]])

    # trace plot
    alpha_line.set_data(range(frame+1), chain[:frame+1, 0])
    beta_line.set_data( range(frame+1), chain[:frame+1, 1])

    status = "accepted" if accepted_list[frame] else "rejected"
    status_text.set_text(f"step {frame+1}  —  {status}")

    return chain_line, chain_dot, leap_line, leap_dot, alpha_line, beta_line, status_text


anim = animation.FuncAnimation(fig, update, frames=n_animate, interval=80, blit=True)
plt.show()

out = f"{fig_dir}/hmc_animation.gif"
print("rendering animation... (this takes a minute)")
#anim.save(out, writer="pillow", fps=12)
plt.close()
print(f"saved {out}")
