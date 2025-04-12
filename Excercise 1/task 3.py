import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial condition
h = 0.01
t0, t_end = 0, 10
steps = int((t_end - t0) / h)

# Initialize arrays
x = np.zeros(steps + 1)
y = np.zeros(steps + 1)
z = np.zeros(steps + 1)
t = np.linspace(t0, t_end, steps + 1)

# Set initial values
x[0], y[0], z[0] = 1.0, 0.0, 0.0

# Euler integration loop
for i in range(steps):
    dx = sigma * (y[i] - x[i])
    dy = x[i] * (rho - z[i]) - y[i]
    dz = x[i] * y[i] - beta * z[i]

    x[i + 1] = x[i] + h * dx
    y[i + 1] = y[i] + h * dy
    z[i + 1] = z[i] + h * dz

# Plotting the 3D trajectory
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.7)
ax.set_title("Lorenz System Simulated with Euler Method")
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.set_zlabel("z(t)")
plt.tight_layout()
plt.show()
