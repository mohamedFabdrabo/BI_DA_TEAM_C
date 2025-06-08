import numpy as np
import matplotlib.pyplot as plt

# --- Transition Matrices ---

# Original transition matrix (P)
P = np.array([
    [0.5, 0.25, 0.25],
    [0.25, 0.5, 0.25],
    [0.25, 0.25, 0.5]
])

# Modified transition matrix (\tilde{P})
P_tilde = np.array([
    [0.5,   0.25,  0.25],
    [0.25,  0.5,   0.25],
    [0.125, 0.125, 0.75]
])

# Target stationary distribution for \tilde{P}
p_star = np.array([0.25, 0.25, 0.5])
p_stationary_P = np.array([1/3, 1/3, 1/3])

# Initial distribution (start in state 1)
pi_0 = np.array([1.0, 0.0, 0.0])

# Step counts to simulate
step_counts = [1, 100, 1000]

# --- Prepare Subplots ---
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle("Markov Chain Distribution Evolution for P and $\\tilde{P}$", fontsize=16)

# --- Simulation Loop ---
for col_idx, max_step in enumerate(step_counts):
    steps = np.arange(max_step + 1)

    # Simulate for P
    dist_P = [pi_0]
    pi = pi_0.copy()
    for _ in range(max_step):
        pi = pi @ P
        dist_P.append(pi.copy())
    dist_P = np.array(dist_P)

    # Plot for P
    ax = axes[0, col_idx]
    ax.plot(steps, dist_P[:, 0], label='State 1')
    ax.plot(steps, dist_P[:, 1], label='State 2')
    ax.plot(steps, dist_P[:, 2], label='State 3')
    for val in p_stationary_P:
        ax.axhline(y=val, linestyle='dashed', color='gray')
    ax.set_title(f'P: Steps = {max_step}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Probability')
    ax.grid(True)
    if col_idx == 0:
        ax.legend()

    # Simulate for \tilde{P}
    dist_P_tilde = [pi_0]
    pi = pi_0.copy()
    for _ in range(max_step):
        pi = pi @ P_tilde
        dist_P_tilde.append(pi.copy())
    dist_P_tilde = np.array(dist_P_tilde)

    # Plot for \tilde{P}
    ax = axes[1, col_idx]
    ax.plot(steps, dist_P_tilde[:, 0], label='State 1')
    ax.plot(steps, dist_P_tilde[:, 1], label='State 2')
    ax.plot(steps, dist_P_tilde[:, 2], label='State 3')
    for val in p_star:
        ax.axhline(y=val, linestyle='dashed', color='gray')
    ax.set_title(f'$\\tilde{{P}}$: Steps = {max_step}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Probability')
    ax.grid(True)
    if col_idx == 0:
        ax.legend()

# --- Final Layout ---
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
