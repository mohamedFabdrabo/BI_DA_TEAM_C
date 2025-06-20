import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def particle_filter(N, y_obs=2):
    # Step (a): Sample X0 and noise
    X0 = np.random.normal(-1, np.sqrt(2), N)
    Xi = np.random.normal(0, 1, N)
    X1 = 0.5 * X0 + 1 + Xi

    # Step (c): Likelihood weights
    w_tilde = norm.pdf(y_obs, loc=X1, scale=np.sqrt(2))

    # Step (d): Normalize weights
    w = w_tilde / np.sum(w_tilde)

    # Step (e): Posterior mean and variance
    m_bar = np.sum(w * X1)
    V_bar = np.sum(w * (X1 - m_bar)**2)

    # Step (f): Effective sample size
    ESS = 1 / np.sum(w**2)

    return X1, m_bar, V_bar, ESS


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, N in enumerate([100, 1000, 10000]):
    X1, m, v, ess = particle_filter(N)

    # Plot histogram
    ax = axes[idx]
    ax.hist(X1, bins=50, density=True, alpha=0.6, label=f'N={N}')
    x = np.linspace(-3, 4, 1000)
    ax.plot(x, norm.pdf(x, loc=0.5, scale=np.sqrt(1.5)), 'r--', label='Theoretical N(0.5, 1.5)')
    ax.set_title(f'N={N}\nMean={m:.2f}, Var={v:.2f}, ESS={ess:.0f}')
    ax.legend()
    ax.grid()

plt.suptitle('Histograms of $X_1$ for Different Particle Sizes')
plt.tight_layout()
plt.show()
