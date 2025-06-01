import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta0 = 0.1
T = 1.0
d = 2
M = 1000

# Exact Sampling
s = 0.2
Sigma_s = (1 + 2 * beta0 * (T - s)) * np.eye(d)
exact_samples = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma_s, size=M)

# Euler-Maruyama Sampling
N_steps = 100
delta_s = T / N_steps
s_grid = np.linspace(0, T, N_steps + 1)
x_em = np.random.multivariate_normal(mean=np.zeros(d), cov=(1 + 2 * beta0 * T) * np.eye(d), size=M)

for i in range(N_steps):
    s = s_grid[i]
    Sigma_inv = 1 / (1 + 2 * beta0 * (T - s))
    noise = np.random.randn(M, d)
    drift = 2 * beta0 * Sigma_inv * x_em
    diffusion = np.sqrt(2 * beta0 * delta_s) * noise
    x_em = x_em + drift * delta_s + diffusion

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(exact_samples[:, 0], exact_samples[:, 1], alpha=0.5, s=10)
plt.title("Exact Sampling at s=0.2")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.scatter(x_em[:, 0], x_em[:, 1], alpha=0.5, s=10, color='orange')
plt.title("Euler-Maruyama Sampling (Reversed)")
plt.axis("equal")

plt.tight_layout()
plt.show()
