import numpy as np
import os

model     = "exponential"
run_id    = "E10_S11"
engine_id = 10
samp_dir  = f"results/{model}/{run_id}/samples"

# reuse the same normalised arrays from the linear model run
# cycles: raw / global_max_cycle (362). y_obs: s11 normalised to [0,1]
cycles = np.load(f"results/linear/{run_id}/samples/cycles_engine{engine_id}.npy")
y_obs  = np.load(f"results/linear/{run_id}/samples/y_obs_engine{engine_id}.npy")

# reparametrisation: sample phi=log(beta) and psi=log(gamma) instead of beta/gamma directly
# this keeps beta and gamma positive by construction (exp() is always > 0)
# and puts all three params on a more comparable scale so a single epsilon works

# priors on alpha — same gaussian as before
mu_alpha    = 0.2868
sigma_alpha = 0.0779

# priors on phi = log(beta): log-normal, derived from fit_exp_priors.py values
# mu_phi = log(mu_beta), sigma_phi ~ sigma_beta / mu_beta (delta method approximation)
mu_phi    = np.log(0.0116)          # ≈ -4.455
sigma_phi = 0.0051 / 0.0116        # ≈ 0.440

# priors on psi = log(gamma): same idea
# mu_psi = log(mu_gamma), sigma_psi ~ sigma_gamma / mu_gamma
mu_psi    = np.log(7.1597)          # ≈ 1.968
sigma_psi = 1.5823 / 7.1597        # ≈ 0.221

sigma_noise = 0.15

# hyperparams — to be tuned
epsilon   = 0.015
L         = 15
n_samples = 10000
burn_in   = 1000


def log_posterior(alpha, phi, psi):
    # recover beta and gamma from the log-space params
    beta  = np.exp(phi)
    gamma = np.exp(psi)

    # likelihood: y_t ~ N(alpha + beta*exp(gamma*t), sigma^2)
    h  = alpha + beta * np.exp(np.clip(gamma * cycles, -500, 500))
    ll = -0.5 * np.sum(((y_obs - h) / sigma_noise) ** 2)

    # gaussian priors on alpha, phi, psi (phi and psi are log-normal priors on beta, gamma)
    lp = (-0.5 * ((alpha - mu_alpha) / sigma_alpha) ** 2
        - 0.5 * ((phi   - mu_phi)   / sigma_phi)   ** 2
        - 0.5 * ((psi   - mu_psi)   / sigma_psi)   ** 2)

    return ll + lp


def compute_gradients(alpha, phi, psi):
    # gradients of log_posterior w.r.t (alpha, phi, psi)
    # chain rule: d/dphi = d/dbeta * dbeta/dphi = d/dbeta * beta
    #             d/dpsi = d/dgamma * dgamma/dpsi = d/dgamma * gamma

    beta   = np.exp(phi)
    gamma  = np.exp(psi)
    exp_gt = np.exp(np.clip(gamma * cycles, -500, 500))
    h      = alpha + beta * exp_gt
    r      = y_obs - h

    # d/dalpha: h changes by 1 per unit alpha
    g_alpha = (1 / sigma_noise**2) * np.sum(r) \
            - (alpha - mu_alpha) / sigma_alpha**2

    # d/dphi: chain rule through beta=exp(phi) — multiply beta gradient by beta
    g_phi   = (1 / sigma_noise**2) * np.sum(r * exp_gt) * beta \
            - (phi - mu_phi) / sigma_phi**2

    # d/dpsi: chain rule through gamma=exp(psi) — multiply gamma gradient by gamma
    g_psi   = (1 / sigma_noise**2) * np.sum(r * beta * cycles * exp_gt) * gamma \
            - (psi - mu_psi) / sigma_psi**2

    # clip gradient magnitude — stops leapfrog exploding on steep parts of the landscape
    return np.clip(np.array([g_alpha, g_phi, g_psi]), -50, 50)


def leapfrog(theta, p):
    # same leapfrog as before, theta is now (alpha, phi, psi)
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
    # start at prior means in log space
    theta = np.array([mu_alpha, mu_phi, mu_psi])

    samples      = np.zeros((n_samples, 3))
    accepted     = 0
    delta_H_vals = []

    for i in range(n_samples):
        p = np.random.randn(3)

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


print("running exponential hmc sampler (reparametrised: phi=log(beta), psi=log(gamma))...")
print(f"engine {engine_id}  |  {len(cycles)} cycles  |  epsilon={epsilon}  L={L}  n_samples={n_samples}")
samples, acc, delta_H_vals = hmc_sampler()

print(f"done. acceptance rate: {acc:.2%}")
print(f"target is 60-80% — {'ok' if 0.6 <= acc <= 0.8 else 'needs tuning'}")

post  = samples[burn_in:]
a_s   = post[:, 0]
phi_s = post[:, 1]
psi_s = post[:, 2]

# recover original params for interpretation
beta_s  = np.exp(phi_s)
gamma_s = np.exp(psi_s)

print(f"\nalpha — mean: {a_s.mean():.4f}    std: {a_s.std():.4f}")
print(f"beta  — mean: {beta_s.mean():.4f}    std: {beta_s.std():.4f}   (recovered from phi)")
print(f"gamma — mean: {gamma_s.mean():.4f}    std: {gamma_s.std():.4f}   (recovered from psi)")
print(f"delta_H — mean: {delta_H_vals.mean():.4f}  (want near 0)")

os.makedirs(samp_dir, exist_ok=True)
# save raw (alpha, phi, psi) samples — convert back to beta/gamma when computing RUL
np.save(f"{samp_dir}/hmc_samples_engine{engine_id}.npy", samples)
np.save(f"{samp_dir}/delta_H_engine{engine_id}.npy", delta_H_vals)
print(f"samples saved to {samp_dir}")
