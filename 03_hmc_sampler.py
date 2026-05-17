import numpy as np
import os

model     = "linear"
run_id    = "E10_S11"
engine_id = 10
samp_dir  = f"results/{model}/{run_id}/samples"

cycles = np.load(f"{samp_dir}/cycles_engine{engine_id}.npy")
y_obs  = np.load(f"{samp_dir}/y_obs_engine{engine_id}.npy")

# gaussian priors on (alpha, beta) derived from OLS across 100 training engines
# alpha: s11 intercept in normalised coords (~0.21 on average)
# beta:  degradation rate — positive because s11 increases with cycle
mu_alpha    =  0.21
sigma_alpha =  0.15
mu_beta     =  0.73
sigma_beta  =  0.25
sigma_noise =  0.15   # sensor noise on normalised scale (fixed, not inferred)

# hmc hyperparams — tuned on training engine 10
epsilon   = 0.015  # leapfrog step size
L         = 20     # leapfrog steps per iteration
n_samples = 10000
burn_in   = 1000


def log_likelihood(alpha, beta, y_obs, cycles, sigma):
    # log N(y_t | alpha + beta*t, sigma^2) summed over all t
    # = -0.5 * sum_t [(y_t - (alpha + beta*t))^2 / sigma^2]
    mu_pred = alpha + beta * cycles
    return -0.5 * np.sum(((y_obs - mu_pred) / sigma) ** 2)


def log_prior(alpha, beta):
    # log N(alpha | mu_alpha, sigma_alpha^2) + log N(beta | mu_beta, sigma_beta^2)
    lp_alpha = -0.5 * ((alpha - mu_alpha) / sigma_alpha) ** 2
    lp_beta  = -0.5 * ((beta  - mu_beta)  / sigma_beta)  ** 2
    return lp_alpha + lp_beta


def log_posterior(alpha, beta):
    # log P(alpha, beta | D) = log P(D | alpha, beta) + log P(alpha, beta)
    # P(D) cancels exactly in the metropolis acceptance ratio — never computed
    return log_likelihood(alpha, beta, y_obs, cycles, sigma_noise) + log_prior(alpha, beta)


def compute_gradients(alpha, beta):
    # analytical grad of log_posterior w.r.t (alpha, beta)
    # this acts as the physical force in the hamiltonian system, steering the sampler
    # toward high-probability regions of the posterior
    #
    # let r_t = y_t - (alpha + beta*t)  (residual at cycle t)
    #
    # d/d_alpha = (1/sigma^2) * sum_t(r_t)        - (alpha - mu_alpha) / sigma_alpha^2
    # d/d_beta  = (1/sigma^2) * sum_t(t * r_t)    - (beta  - mu_beta)  / sigma_beta^2
    residuals  = y_obs - (alpha + beta * cycles)
    grad_alpha = (1 / sigma_noise**2) * np.sum(residuals) \
                 - (alpha - mu_alpha) / sigma_alpha**2
    grad_beta  = (1 / sigma_noise**2) * np.sum(cycles * residuals) \
                 - (beta - mu_beta) / sigma_beta**2
    return np.array([grad_alpha, grad_beta])


def leapfrog(theta, p, epsilon, L):
    # discretises hamilton's equations: dtheta/dt = p, dp/dt = grad log P(theta|D)
    # staggered half-step momentum + full-step position = second order accuracy (vs euler)
    # keeps the hamiltonian H = -log P(theta|D) + p^2/2 approximately conserved
    theta = theta.copy()
    p     = p.copy()

    p += (epsilon / 2) * compute_gradients(theta[0], theta[1])   # half step

    for _ in range(L - 1):
        theta += epsilon * p                                        # full position step
        p     += epsilon * compute_gradients(theta[0], theta[1])   # full momentum step

    theta += epsilon * p                                            # last position step
    p     += (epsilon / 2) * compute_gradients(theta[0], theta[1]) # last half momentum step

    return theta, p


def hmc_sampler():
    # hmc loop:
    #   1. sample fresh momentum p ~ N(0, I)  — this randomises the direction each iteration
    #   2. run leapfrog for L steps to get a proposal (theta_new, p_new)
    #   3. metropolis accept/reject based on delta_H = H_old - H_new
    #      accept if delta_H > 0 (lower energy = higher posterior), else accept with prob exp(delta_H)
    #      perfect leapfrog would give delta_H = 0 always — small epsilon keeps it near 0
    #   4. accepted proposals = new sample. rejected = repeat current position
    theta        = np.array([mu_alpha, mu_beta])   # start at prior means
    samples      = np.zeros((n_samples, 2))
    accepted     = 0
    delta_H_vals = []

    for i in range(n_samples):
        p          = np.random.randn(2)
        H_current  = -log_posterior(theta[0], theta[1]) + 0.5 * np.dot(p, p)
        th_p, p_p  = leapfrog(theta, p, epsilon, L)
        H_proposed = -log_posterior(th_p[0], th_p[1]) + 0.5 * np.dot(p_p, p_p)

        delta_H = H_current - H_proposed
        delta_H_vals.append(delta_H)

        if np.log(np.random.rand()) < delta_H:
            theta = th_p
            accepted += 1

        samples[i] = theta

    return samples, accepted / n_samples, np.array(delta_H_vals)


samples, acceptance_rate, delta_H_vals = hmc_sampler()

print(f"acceptance rate: {acceptance_rate:.2%}  (target 60-80% — {'ok' if 0.6 <= acceptance_rate <= 0.8 else 'needs tuning'})")

post_samples  = samples[burn_in:]
alpha_samples = post_samples[:, 0]
beta_samples  = post_samples[:, 1]

print(f"alpha — mean: {alpha_samples.mean():.4f}  std: {alpha_samples.std():.4f}")
print(f"beta  — mean: {beta_samples.mean():.4f}  std: {beta_samples.std():.4f}")
print(f"delta_H mean: {delta_H_vals.mean():.4f}  (near 0 = leapfrog conserving energy ok)")

os.makedirs(samp_dir, exist_ok=True)
np.save(f"{samp_dir}/hmc_samples_engine{engine_id}.npy", samples)
np.save(f"{samp_dir}/delta_H_engine{engine_id}.npy", delta_H_vals)
