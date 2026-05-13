import numpy as np
import os

# change this to switch runs - must match what 02_preprocessing.py used
run_id   = "E1_S11"
samp_dir = f"results/{run_id}/samples"

cycles = np.load(f"{samp_dir}/cycles_engine1.npy")
y_obs  = np.load(f"{samp_dir}/y_obs_engine1.npy")

#  priors
# these are on the normalised scale (y_obs is in [0,1], mean ~0.37)
# alpha: where s11 starts. mean of y_obs is ~0.37 so prior centres there
# beta: how fast s11 changes per cycle. s11 increases with degradation so beta is positive
# sigma: sensor noise on the normalised scale
mu_alpha    =  0.1    # engine starts near 0 on normalised scale (fresh = low s11)
sigma_alpha =  0.2
mu_beta     =  0.6    # s11 rises ~0.6 units over normalised cycle range
sigma_beta  =  0.3
sigma_noise =  0.15   # sensor noise on normalised scale

# hmc hyperparameters
epsilon  = 0.014   # tuned to hit ~75% acceptance rate
L        = 20      # leapfrog steps per iteration
n_samples = 10000
burn_in   = 1000


def log_likelihood(alpha, beta, y_obs, cycles, sigma):
    # gaussian log-likelihood: how well do (alpha, beta) explain the sensor readings
    # sum of log N(y_t | alpha + beta*t, sigma) over all cycles
    mu_pred = alpha + beta * cycles
    return -0.5 * np.sum(((y_obs - mu_pred) / sigma) ** 2)


def log_prior(alpha, beta):
    # gaussian priors on both params
    # penalises values far from our prior beliefs about alpha and beta
    lp_alpha = -0.5 * ((alpha - mu_alpha) / sigma_alpha) ** 2
    lp_beta  = -0.5 * ((beta  - mu_beta)  / sigma_beta)  ** 2
    return lp_alpha + lp_beta


def log_posterior(alpha, beta):
    # the thing hmc navigates - just likelihood + prior in log space
    # p(D) cancels in the acceptance ratio so we never compute it
    return log_likelihood(alpha, beta, y_obs, cycles, sigma_noise) + log_prior(alpha, beta)


def compute_gradients(alpha, beta):
    # analytical gradients of log_posterior w.r.t. alpha and beta
    # these act as the force that drives the leapfrog integrator
    residuals = y_obs - (alpha + beta * cycles)

    grad_alpha = (1 / sigma_noise**2) * np.sum(residuals) \
                 - (alpha - mu_alpha) / sigma_alpha**2

    grad_beta  = (1 / sigma_noise**2) * np.sum(cycles * residuals) \
                 - (beta - mu_beta) / sigma_beta**2

    return np.array([grad_alpha, grad_beta])


def leapfrog(theta, p, epsilon, L):
    # simulates a ball rolling across the posterior landscape for L steps
    # staggered half-step momentum, full-step position keeps energy ~conserved
    theta = theta.copy()
    p = p.copy()

    # half step for momentum at the start
    p += (epsilon / 2) * compute_gradients(theta[0], theta[1])

    for _ in range(L - 1):
        theta += epsilon * p                                        # full position step
        p     += epsilon * compute_gradients(theta[0], theta[1])   # full momentum step

    theta += epsilon * p                                            # final position step
    p     += (epsilon / 2) * compute_gradients(theta[0], theta[1]) # final half momentum step

    return theta, p


def hmc_sampler():
    # main loop - each iteration proposes a new (alpha, beta) via leapfrog
    # then accepts or rejects it based on how much energy changed

    # start at prior means - reasonable starting point
    theta = np.array([mu_alpha, mu_beta])

    samples      = np.zeros((n_samples, 2))
    accepted     = 0
    delta_H_vals = []   # track energy change each step - should stay near 0

    for i in range(n_samples):
        # draw fresh momentum each iteration - this is the hmc trick
        p = np.random.randn(2)

        # current hamiltonian: potential energy + kinetic energy
        # U = -log_posterior (negative bc posterior peak = energy minimum)
        # K = p^2 / 2
        H_current = -log_posterior(theta[0], theta[1]) + 0.5 * np.dot(p, p)

        # propose new (theta, p) by running leapfrog
        theta_proposed, p_proposed = leapfrog(theta, p, epsilon, L)

        H_proposed = -log_posterior(theta_proposed[0], theta_proposed[1]) + 0.5 * np.dot(p_proposed, p_proposed)

        # metropolis acceptance step
        # if proposed has lower energy (higher posterior) - always accept
        # if higher energy - accept with probability exp(H_current - H_proposed)
        delta_H = H_current - H_proposed
        delta_H_vals.append(delta_H)

        if np.log(np.random.rand()) < delta_H:
            theta = theta_proposed
            accepted += 1

        samples[i] = theta

    acceptance_rate = accepted / n_samples
    return samples, acceptance_rate, np.array(delta_H_vals)


print("running hmc sampler...")
samples, acceptance_rate, delta_H_vals = hmc_sampler()

print(f"done. acceptance rate: {acceptance_rate:.2%}")
print(f"target is 60-80% - {'ok' if 0.6 <= acceptance_rate <= 0.8 else 'needs tuning'}")

# quick look at posterior
post_samples = samples[burn_in:]
alpha_samples = post_samples[:, 0]
beta_samples  = post_samples[:, 1]

print(f"\nalpha — mean: {alpha_samples.mean():.4f}  std: {alpha_samples.std():.4f}")
print(f"beta  — mean: {beta_samples.mean():.4f}  std: {beta_samples.std():.4f}")
print(f"delta_H — mean: {delta_H_vals.mean():.4f}  (should be near 0)")

os.makedirs(samp_dir, exist_ok=True)
np.save(f"{samp_dir}/hmc_samples_engine1.npy", samples)
np.save(f"{samp_dir}/delta_H_engine1.npy", delta_H_vals)
print(f"samples saved to {samp_dir}")
