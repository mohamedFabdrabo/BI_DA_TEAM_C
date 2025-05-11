import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the function and true integral value
f = lambda x: np.exp(x)
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
        contrib = h * (w1 * f(mapped_x1) + w2 * f(mapped_x2) + w3 * f(mapped_x3))
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