import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Parameters for Lorenz-63
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Right-hand side of Lorenz system
def lorenz(state):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

# Exercise 1: integrate with modified Euler (midpoint) and record every 5 steps
def integrate_lorenz_midpoint(x0, h=0.01, t_end=200.0, record_every=3):
    n_steps = int(t_end / h)
    m = n_steps // record_every + 1
    traj = np.zeros((3, m))
    times = np.zeros(m)
    x = np.array(x0, dtype=float)
    traj[:, 0] = x
    times[0] = 0.0
    for n in range(1, n_steps + 1):
        g1 = lorenz(x)
        x_mid = x + 0.5 * h * g1
        g2 = lorenz(x_mid)
        x = x + h * g2
        if n % record_every == 0:
            idx = n // record_every
            traj[:, idx] = x
            times[idx] = n * h
    return times, traj

# Exercise 2: observe only x-coordinate at each output time (excluding t=0)
def make_observations(times, traj):
    obs_times = times[1:]
    obs_values = traj[0, 1:]
    return obs_times, obs_values

# Exercise 3: linear extrapolation forecasts and RMSE
def forecast_rmse(obs_values, lead_steps):
    errors = []
    N = len(obs_values)
    for k in range(1, N - lead_steps):
        prev = obs_values[k - 1]
        curr = obs_values[k]
        forecast = curr + lead_steps * (curr - prev)
        truth = obs_values[k + lead_steps]
        errors.append((forecast - truth) ** 2)
    return np.sqrt(np.mean(errors))

if __name__ == '__main__':
    # Determine directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Initial condition
    x0 = [1.0, 1.0, 1.0]

    # Integrate and store trajectory in human-readable CSV
    times, traj = integrate_lorenz_midpoint(x0)
    data_ref = np.vstack((times, traj))
    ref_path = os.path.join(script_dir, 'reference_traj.csv')
    np.savetxt(ref_path, data_ref.T, delimiter=',',
               header='time,x,y,z', comments='')
    print(f'Saved reference trajectory to {ref_path}')

    # Plot Lorenz attractor to verify reproduction
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[0], traj[1], traj[2], lw=0.5)
    ax.set_title('Lorenz Attractor (Midpoint, h=0.01)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.show()
    
    # Generate and save observations
    obs_times, obs_values = make_observations(times, traj)
    data_obs = np.vstack((obs_times, obs_values))
    obs_path = os.path.join(script_dir, 'observations.csv')
    np.savetxt(obs_path, data_obs.T, delimiter=',',
               header='time,x_obs', comments='')
    print(f'Saved observations to {obs_path}')

    # Plot first 200 observed x-values as markers (Exercise 2)
    plt.figure(figsize=(8, 4))
    # Scatter only first 4000 points (0 ≤ time ≤ 10)
    n_plot = 4000
    plt.scatter(obs_times, obs_values, marker = 'x',  s=10)
    plt.title('observed values')
    plt.xlabel('time')
    plt.ylabel('x-coordinate')
    plt.xlim(50, 75)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute RMSE for lead times of 1 and 3 output intervals
    rmse_1 = forecast_rmse(obs_values, lead_steps=1)
    rmse_3 = forecast_rmse(obs_values, lead_steps=3)

    print(f'RMSE for Δt = 0.05: {rmse_1:.6f}')
    print(f'RMSE for 3Δt = 0.15: {rmse_3:.6f}')
