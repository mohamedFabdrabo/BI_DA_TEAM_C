import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, y0, t_span, h):
    t = np.arange(t_span[0], t_span[1] + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + h * f(y[i], t[i])
    return t, y

# Define the ODE dy/dt = -2y
f = lambda y, t: -2 * y

# Using step-size h = 0.1
t1, y1 = euler_method(f, 1, (0, 1), 0.1)

# Using step-size h = 0.01
t2, y2 = euler_method(f, 1, (0, 1), 0.01)

# Exact solution
exact_sol = lambda t: np.exp(-2 * t)
t_exact = np.linspace(0, 1, 100)
y_exact = exact_sol(t_exact)

# Plot the solutions
plt.figure(figsize=(8, 5))
plt.plot(t1, y1, 'ro-', label='Euler, h = 0.1')
plt.plot(t2, y2, 'bo-', label='Euler, h = 0.01')
plt.plot(t_exact, y_exact, 'k-', label='Exact solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler Method Approximations vs. Exact Solution')
plt.legend()
plt.grid(True)
plt.show()

# Compute and print errors at t = 1
error_h1 = abs(y1[-1] - exact_sol(1))
error_h2 = abs(y2[-1] - exact_sol(1))
print("Error at t = 1 for h = 0.1: ", error_h1)
print("Error at t = 1 for h = 0.01: ", error_h2)