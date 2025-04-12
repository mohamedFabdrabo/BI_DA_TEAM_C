import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euler_lorenz(initial_state, t_max, h):
    """
    Solves the Lorenz system using Euler's method.

    Parameters:
        initial_state: array-like, initial values [x0, y0, z0]
        t_max: float, maximum time for simulation
        h: float, time step size

    Returns:
        t: numpy array of time points
        traj: numpy array of shape (num_steps, 3) with state trajectories
    """
    num_steps = int(t_max / h) + 1
    t = np.linspace(0, t_max, num_steps)
    traj = np.zeros((num_steps, 3))
    traj[0] = initial_state
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    for i in range(num_steps - 1):
        x, y, z = traj[i]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        traj[i + 1] = traj[i] + h * np.array([dx, dy, dz])
    return t, traj

# Simulation parameters
initial_state = np.array([1, 0, 0])
t_max = 10
h = 0.01

# Compute the numerical trajectory for the Lorenz system
t, traj = euler_lorenz(initial_state, t_max, h)

# Plotting the 3D trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.8)
ax.set_title("Lorenz System Trajectory via Euler Method")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()