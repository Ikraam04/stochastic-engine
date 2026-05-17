import numpy as np
import os

model     = "exponential"
run_id    = "E10_S11"
engine_id = 10
samp_dir  = f"results/{model}/{run_id}/samples"

# reuse normalised arrays from the linear run — same normalisation scheme
cycles = np.load(f"results/linear/{run_id}/samples/cycles_engine{engine_id}.npy")
y_obs  = np.load(f"results/linear/{run_id}/samples/y_obs_engine{engine_id}.npy")

# reparametrisation: sample phi=log(beta) and psi=log(gamma) instead of beta and gamma directly
# why bother:
#   1. positivity: beta > 0 and gamma > 0 are hard constraints.
#      sampling phi/psi and recovering beta=exp(phi), gamma=exp(psi) enforces this automatically
#   2. scale: beta ~ 0.01 and gamma ~ 7, so direct sampling needs different step sizes per param
#      in log-space all three params are closer in scale so one epsilon works
#
# priors on alpha — same gaussian as the linear model
mu_alpha    = 0.2868
sigma_alpha = 0.0779

# priors on phi = log(beta), derived using the delta method:
# if X ~ N(mu, sigma^2) then log(X) is approximately N(log(mu), (sigma/mu)^2)
# this is a first-order taylor expansion of log() around the mean
mu_phi    = np.log(0.0116)    # = log(mu_beta)
sigma_phi = 0.0051 / 0.0116  # = sigma_beta / mu_beta

# same for psi = log(gamma)
mu_psi    = np.log(7.1597)
sigma_psi = 1.5823 / 7.1597

sigma_noise = 0.15

epsilon   = 0.015
L         = 15
n_samples = 10000
burn_in   = 1000


def log_posterior(alpha, phi, psi):
    # recover beta and gamma from log-space — always positive by construction
    beta  = np.exp(phi)
    gamma = np.exp(psi)

    # likelihood: y_t ~ N(h(t), sigma^2) where h(t) = alpha + beta*exp(gamma*t)
    # clip gamma*t to avoid overflow — long engines near failure get very large exp(gamma*t)
    h  = alpha + beta * np.exp(np.clip(gamma * cycles, -500, 500))
    ll = -0.5 * np.sum(((y_obs - h) / sigma_noise) ** 2)

    # gaussian prior in log-space = log-normal prior on beta and gamma
    lp = (-0.5 * ((alpha - mu_alpha) / sigma_alpha) ** 2
        - 0.5 * ((phi   - mu_phi)   / sigma_phi)   ** 2
        - 0.5 * ((psi   - mu_psi)   / sigma_psi)   ** 2)

    return ll + lp


def compute_gradients(alpha, phi, psi):
    # analytical grad of log_posterior w.r.t (alpha, phi, psi)
    #
    # for phi and psi we need the chain rule since we're in log-space:
    #   d/d_phi  = d/d_beta  * d_beta/d_phi  = d/d_beta  * beta   (d exp(phi)/d phi = exp(phi) = beta)
    #   d/d_psi  = d/d_gamma * d_gamma/d_psi = d/d_gamma * gamma
    #
    # let r_t = y_t - h(t)  (residual)
    #
    # d/d_alpha  = (1/sigma^2) * sum_t(r_t)
    # d/d_beta   = (1/sigma^2) * sum_t(r_t * exp(gamma*t))
    # d/d_gamma  = (1/sigma^2) * sum_t(r_t * beta * t * exp(gamma*t))
    #
    # then multiply d/d_beta by beta, d/d_gamma by gamma for the chain rule:
    # d/d_phi    = (1/sigma^2) * sum_t(r_t * exp(gamma*t)) * beta
    # d/d_psi    = (1/sigma^2) * sum_t(r_t * beta * t * exp(gamma*t)) * gamma
    beta   = np.exp(phi)
    gamma  = np.exp(psi)
    exp_gt = np.exp(np.clip(gamma * cycles, -500, 500))
    r      = y_obs - (alpha + beta * exp_gt)

    g_alpha = (1 / sigma_noise**2) * np.sum(r) \
            - (alpha - mu_alpha) / sigma_alpha**2

    g_phi   = (1 / sigma_noise**2) * np.sum(r * exp_gt) * beta \
            - (phi - mu_phi) / sigma_phi**2

    g_psi   = (1 / sigma_noise**2) * np.sum(r * beta * cycles * exp_gt) * gamma \
            - (psi - mu_psi) / sigma_psi**2

    # clip gradient magnitude — stops leapfrog exploding on steep regions of the posterior
    # (long engines near failure have very large exp(gamma*t) terms that push gradients huge)
    # clipping stabilises the integrator. it does NOT change the target distribution
    return np.clip(np.array([g_alpha, g_phi, g_psi]), -50, 50)


def leapfrog(theta, p):
    # same staggered leapfrog as linear model — theta is now (alpha, phi, psi)
    theta = theta.copy()
    p     = p.copy()

    p += (epsilon / 2) * compute_gradients(theta[0], theta[1], theta[2])

    for _ in range(L - 1):
        theta += epsilon * p
        p     += epsilon * compute_gradients(theta[0], theta[1], theta[2])

    theta += epsilon * p
    p     += (epsilon / 2) * compute_gradients(theta[0], theta[1], theta[2])

    return theta, p


def hmc_sampler():
    theta        = np.array([mu_alpha, mu_phi, mu_psi])   # start at prior means in log-space
    samples      = np.zeros((n_samples, 3))
    accepted     = 0
    delta_H_vals = []

    for i in range(n_samples):
        p         = np.random.randn(3)
        H_cur     = -log_posterior(*theta) + 0.5 * np.dot(p, p)
        th_p, p_p = leapfrog(theta, p)
        H_prop    = -log_posterior(*th_p)  + 0.5 * np.dot(p_p, p_p)

        delta_H = H_cur - H_prop
        delta_H_vals.append(delta_H)

        if np.log(np.random.rand()) < delta_H:
            theta    = th_p
            accepted += 1

        samples[i] = theta

    return samples, accepted / n_samples, np.array(delta_H_vals)


samples, acc, delta_H_vals = hmc_sampler()

print(f"acceptance rate: {acc:.2%}  (target 60-80% — {'ok' if 0.6 <= acc <= 0.8 else 'needs tuning'})")

post    = samples[burn_in:]
a_s     = post[:, 0]
phi_s   = post[:, 1]
psi_s   = post[:, 2]
beta_s  = np.exp(phi_s)
gamma_s = np.exp(psi_s)

print(f"alpha — mean: {a_s.mean():.4f}    std: {a_s.std():.4f}")
print(f"beta  — mean: {beta_s.mean():.4f}    std: {beta_s.std():.4f}   (recovered from phi)")
print(f"gamma — mean: {gamma_s.mean():.4f}    std: {gamma_s.std():.4f}   (recovered from psi)")
print(f"delta_H mean: {delta_H_vals.mean():.4f}  (near 0 = leapfrog conserving energy ok)")

os.makedirs(samp_dir, exist_ok=True)
# store raw (alpha, phi, psi) — recover beta/gamma with exp() when computing RUL in 05
np.save(f"{samp_dir}/hmc_samples_engine{engine_id}.npy", samples)
np.save(f"{samp_dir}/delta_H_engine{engine_id}.npy", delta_H_vals)
