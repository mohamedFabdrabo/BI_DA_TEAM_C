from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Exercise 2: Euler Method for dy/dt = -2y

# Part (a): Implement Euler Method

"""  Fixed parameters for this specific problem:
        - f(t, y) = -2y (hardcoded derivative function)
        - t0 = 0 (initial time)
        - t_end = 1 (end time)
        - y0 = 1 (initial condition)
"""

def euler_method(f, y0, t0, t_end, h):
    """
    Euler method implementation
    
    Returns:
    t : array : time values
    y : array : numerical solution
    """
    
    # Create time array from t0 to t_end with step size h
    # We add h to t_end in arange to ensure we include the endpoint
    t = np.arange(t0, t_end + h, h)
    
    y = np.zeros(len(t)) # Initialize solution array with zeros
    y[0] = y0 # Set initial condition
    
    # Euler iteration loop
    for i in range(1, len(t)):
        # Calculate next value using Euler formula:
        # y_new = y_old + h * f(t_old, y_old)
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])
    return t, y

f = lambda t, y: -2 * y
y0 = 1
t0, t_end = 0, 1

t_01, y_01 = euler_method(f, y0, t0, t_end, 0.1)
t_001, y_001 = euler_method(f, y0, t0, t_end, 0.01)

# Part (b): Plot Solutions

# Generate exact solution
exact_t = np.linspace(0, 1, 1000)
exact_y = np.exp(-2 * exact_t)  # y(t) = e^(-2t)

# Plot all solutions
plt.figure(figsize=(10, 6))
plt.plot(t_01, y_01, 'bo--', markersize=6, label='Euler h=0.1')
plt.plot(t_001, y_001, 'g--', linewidth=1, label='Euler h=0.01')
plt.plot(exact_t, exact_y, 'r-', label='Exact Solution')
plt.title('Numerical vs Exact Solution')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Part (c): Error Analysis

# Calculate errors
exact_t1 = np.exp(-2) # Get exact value at t = 1

# Calculate absolute errors at final time point
error_01 = abs(y_01[-1] - exact_t1)
error_001 = abs(y_001[-1] - exact_t1)

print("Error Analysis:")
print(f'Error at t=1 (h=0.1): {error_01:.6f}')
print(f'Error at t=1 (h=0.01): {error_001:.6f}')
print(f'Error ratio: {error_01/error_001:.2f}')

# Calculate RMSE over entire solution
def calculate_rmse(numerical_t, numerical_y):
    """Calculate root mean square error against exact solution"""
    exact_values = np.exp(-2 * numerical_t)
    return np.sqrt(np.mean((exact_values - numerical_y)**2))

print("\nRMSE Analysis:")
print(f"RMSE (h=0.1):  {calculate_rmse(t_01, y_01):.6f}")
print(f"RMSE (h=0.01): {calculate_rmse(t_001, y_001):.6f}")

# Exercise 3: Lorenz System

# Implement Euler Method for Lorenz System
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
plt.clf() # clearing the previous plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.8)
ax.set_title("Lorenz System Trajectory via Euler Method")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()