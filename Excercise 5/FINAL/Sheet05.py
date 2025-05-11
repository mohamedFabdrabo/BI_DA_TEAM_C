import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Exercise 2 Part b


def eval_MC_mean(M, N):
    # Scale is standard deviation (sqrt of variance)
    x = np.random.normal(1, np.sqrt(3), (M, N))
    f_M = (1 + 2*x + x**2).mean(axis=0)
    return f_M  # Output (N,) array


# Simulate for M = 2^0 to 2^8
tmp = []
for i in range(9):
    M = 2**i
    tmp.append(eval_MC_mean(M, 10000))
result = np.array(tmp)
mean = result.mean(axis=1)
var = result.var(axis=1)  # Uses unbiased variance (ddof=1 for sample variance)

# Plotting
M = [2**i for i in range(9)]
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Mean plot
ax1.plot(M, mean, 'ro', label='Monte Carlo Mean')
# Improved horizontal line
ax1.axhline(7, color='b', linestyle='--', label='Analytical Mean')
ax1.set_xscale('log')  # Better visualization for exponential M
ax1.set_xticks(M)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Show actual M values
ax1.grid()
ax1.set_title("Mean")
ax1.legend()

# Variance plot
ax2.plot(M, var, 'ro', label='Monte Carlo Variance')
ax2.set_xscale('log')
ax2.set_xticks(M)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.grid()
ax2.set_title("Variance")
ax2.legend()

plt.tight_layout()
plt.show()

# Exercise 3 Part c
# Define the function and true integral value


def f(x): return np.exp(x)


true_value = np.e - 1  # True value from 0 to 1

# Quadrature nodes and weights (from part a)
x1, x2, x3 = 0, 0.5, 1  # Quadrature nodes
w1, w2, w3 = 1/6, 2/3, 1/6  # Weights

N_list = [1, 2, 4, 8, 16, 32, 64, 128]

errors = []

for N in N_list:
    h = 1 / N  # Subinterval width
    total = 0.0  # Initialize integral approximation

    # Apply quadrature rule to each subinterval
    for k in range(N):
        a = k * h  # Left endpoint of subinterval
        b = a + h  # Right endpoint of subinterval

        # Map quadrature nodes to current subinterval [a, b]
        mapped_x1 = a + x1 * h  # = a
        mapped_x2 = a + x2 * h  # = a + 0.5h
        mapped_x3 = a + x3 * h  # = b

        # Contribution from current subinterval
        contrib = h * (w1 * f(mapped_x1) + w2 *
                       f(mapped_x2) + w3 * f(mapped_x3))
        total += contrib

    error = abs(total - true_value)  # Calculate error
    print(f"Approximation for N={N}: {total:.10f} | Error: {error:.3e}")
    errors.append(error)


# Plot error vs. N on log-log scale
plt.figure(figsize=(8, 5))
plt.loglog(N_list, errors, 'o-', label='Error')

plt.xscale('log', base=2)
plt.yscale('log')
plt.xticks(N_list, labels=[f"{N}" for N in N_list])  # Show exact N values
plt.xlabel('Number of Subintervals ($N_l$)', fontsize=12)

plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())     # Hide minor ticks
plt.ylabel('Absolute Error', fontsize=12)

# Add grid and title
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.title('Error vs. $N_l$ (Log-Log Scale)', fontsize=14)
plt.legend(fontsize=12)

plt.tight_layout()  # Prevent label overlap
plt.show()
