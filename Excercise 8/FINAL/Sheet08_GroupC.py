import numpy as np
import matplotlib.pyplot as plt

'''
Problem 1:
a) Simulate particles from the SDE up to a final time T = 2.0 using the Euler–Maruyama
method.

b) Compare the histogram of the particle distribution at times t ∈ {0.5, 1.0, 2.0} to the
analytical solution.

c) Briefly discuss whether the empirical distribution matches the analytical solution at each
time point.
'''

# Parameters
T = 2.0
dt = 0.005
n_steps = int(T / dt)
n_particles = 10000
times_to_save = [0.5, 1.0, 2.0]
steps_to_save = [int(t / dt) for t in times_to_save]

# Initialize particles from X0 ~ N(0,1)
x = np.random.normal(0, 1, n_particles)
saved_states = {t: np.zeros(n_particles) for t in times_to_save}

# Euler-Maruyama simulation
for step in range(1, n_steps + 1):
    dW = np.random.normal(0, np.sqrt(dt), n_particles)
    x = x - x * dt + np.sqrt(2) * dW
    t_current = step * dt
    if t_current in times_to_save:
        saved_states[t_current] = x.copy()


# Plot histograms and analytical solutions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, t in enumerate(times_to_save):
    ax = axes[i]
    data = saved_states[t]
    
    # Plot histogram
    ax.hist(data, bins=50, density=True, alpha=0.6, color='b', label='Simulated Histogram')
    
    # Provided analytical solution (for X0=0)
    var_provided = 1 - np.exp(-2 * t)
    x_vals = np.linspace(-4, 4, 400)
    p_provided = (1 / np.sqrt(2 * np.pi * var_provided)) * np.exp(-x_vals**2 / (2 * var_provided))
    ax.plot(x_vals, p_provided, 'g-', linewidth=2, label='Provided Solution')
    
    # Correct analytical solution (for X0 ~ N(0,1))
    p_correct = (1 / np.sqrt(2 * np.pi)) * np.exp(-x_vals**2 / 2)
    ax.plot(x_vals, p_correct, 'r-', linewidth=2, label='Correct Solution')
    
    ax.set_title(f't = {t}')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
#plt.savefig('sde_comparison.png')
plt.show()


'''
Problem 2:

a) Well-specified case: The rank histogram is relatively flat, indicating that the ensemble forecast accurately represents the distribution of the true process. 
There is no significant skewness or shape bias, suggesting that the forecast is both unbiased and well-calibrated.

(b) Misspecified case: The rank histogram exhibits a pronounced U-shape, with a high frequency of ranks at 0 and 100. This suggests that the ensemble forecast consistently underestimates the variability of the true process.
The observation frequently falls outside the ensemble spread, revealing underdispersion and significant model bias due to incorrect parameter choices.
'''

# Parameters
N_obs = 100
M = 100
true_a = 0.8
true_sigma = 1.0
biased_a = 0.3
biased_sigma = 0.5

# Set random seed for reproducibility
np.random.seed(42)

# Simulate the true process:
X_obs = np.zeros(N_obs + 1)
for k in range(N_obs):
    X_obs[k+1] = true_a * X_obs[k] + true_sigma * np.random.randn()

# Function to simulate ensembles
def simulate_ensemble(a, sigma):
    ensemble = np.zeros((M, N_obs + 1))
    for m in range(M):
        for k in range(N_obs):
            ensemble[m, k+1] = a * ensemble[m, k] + sigma * np.random.randn()
    return ensemble

# Simulate ensembles for both well-specified and misspecified cases
ensemble_well = simulate_ensemble(true_a, true_sigma)
ensemble_biased = simulate_ensemble(biased_a, biased_sigma)

# Compute rank histograms
def compute_rank_histogram(ensemble, observation):
    ranks = []
    for k in range(1, N_obs + 1):
        sorted_ens = np.sort(ensemble[:, k])
        rank = np.searchsorted(sorted_ens, observation[k], side='right')
        ranks.append(rank)
    hist, _ = np.histogram(ranks, bins=np.arange(M+2))
    return hist

rank_hist_well = compute_rank_histogram(ensemble_well, X_obs)
rank_hist_biased = compute_rank_histogram(ensemble_biased, X_obs)

# Plotting
def plot_rank_histogram(hist, title):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(hist)), hist, width=1.0, edgecolor='black')
    plt.title(title)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

plot_rank_histogram(rank_hist_well, "Rank Histogram - Well-Specified Forecast")
plot_rank_histogram(rank_hist_biased, "Rank Histogram - Misspecified Forecast")

'''
Problem 3: 

part c) Using the forecasts from Problem 2, compute the empirical CRPS for the well-specified
and misspecified ensembles, and discuss how the scores corroborate the conclusions drawn
from the rank histograms.
'''

def crps_ensemble(y_ens, y_obs):
    M = len(y_ens)
    term1 = np.mean(np.abs(y_ens - y_obs))
    diffs = np.abs(y_ens[:, None] - y_ens[None, :])
    term2 = np.mean(diffs) / 2
    return term1 - term2

def compute_average_crps(ensemble, observations):
    crps_values = []
    for k in range(1, N_obs + 1):
        y_ens = ensemble[:, k]
        y_obs = observations[k]
        crps_k = crps_ensemble(y_ens, y_obs)
        crps_values.append(crps_k)
    return np.mean(crps_values), crps_values

# Compute empirical CRPS for both ensembles using the simulated observations from Problem 2
avg_crps_well, crps_well = compute_average_crps(ensemble_well, X_obs)
avg_crps_biased, crps_biased = compute_average_crps(ensemble_biased, X_obs)

print("\n=========== Empirical CRPS Results ===========")
print(f"Well-specified forecast:   Average CRPS = {avg_crps_well:.4f}")
print(f"Misspecified forecast:     Average CRPS = {avg_crps_biased:.4f}")

# plot CRPS over time to better visualize the differences
plt.figure(figsize=(10, 4))
plt.plot(crps_well, label='Well-specified')
plt.plot(crps_biased, label='Misspecified')
plt.title('CRPS Over Time')
plt.xlabel('Time Step')
plt.ylabel('CRPS')
plt.legend()
plt.grid(True)
plt.show()
