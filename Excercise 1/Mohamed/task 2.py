import numpy as np
import matplotlib.pyplot as plt

# Parameters
y0 = 1
t0 = 0
t_end = 1


# the correct Exact time and solution with time from 0 to 1 with 1000 parts on line space
t_exact = np.linspace(t0, t_end, 1000)
y_exact = np.exp(-2 * t_exact) # y(t) = exp(-2t)

# Plot the exact solution
plt.figure(figsize=(10, 6))

# Loop over both step sizes
for h in [0.1, 0.01]:
    t = np.arange(t0, t_end + h, h)
    y = np.zeros_like(t)
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i - 1] + h * (-2 * y[i - 1])  # dy/dt = -2y

    # Compute error at t = 1
    y_true = np.exp(-2 * t[-1])
    error = abs(y[-1] - y_true)
    print(f"Using step size h = {h} → y(1) ≈ {y[-1]:.6f}, Exact = {y_true:.6f}, Error = {error:.6f}")

    # Plot the numerical solution
    plt.plot(t, y, label=f"Euler method with h={h}")

# plot the exact solution : 
plt.plot(t_exact, y_exact, 'k-', label='Exact: y(t) = exp(-2t)', linewidth=2)

# Finalize plot
plt.title("Euler Method Approximation vs Exact Solution")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()
