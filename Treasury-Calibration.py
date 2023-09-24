import numpy as np
from scipy.optimize import minimize

T = 10  # Total time period
num_periods = 10
dt = T / num_periods
number_paths = 100
market_data = np.array([0.0312, 0.0320, 0.0325, 0.0328, 0.0333, 0.0337, 0.0340, 0.0343, 0.0345, 0.0347])
sigma = np.array([0.15, 0.15, 0.15, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
alpha = 0.1

"""
actual: e^{-0.0312}

model:
    r = r[0] + 0.1 * (theta - r[0]) + 0.15 * N(0, 1)
      = 0.0312 + 0.1 * (theta - 0.0312) + 0.15 * N(0, 1)
      => theta = 0.0312
"""

r = np.zeros((num_periods, number_paths)) # short rate
r[0, :] = 0.0312
bond_prices = np.exp(-market_data)
theta = np.zeros(num_periods-1)

def generate_standard_norm(steps, number_paths):
    """
    100 standard normal for each timestep
    return a 2d numpy (9, 100)
    """
    return np.random.normal(size=(steps, number_paths))

paths = generate_standard_norm(num_periods-1, number_paths)

# Calibrate recursively
for t in [1,2,3,4,5,6,7,8,9]:
    z = paths[t-1]
    def objective_function(theta):
        actual_bond_price = bond_prices[t]
        model_rate = r[t-1] + alpha * (theta - r[t-1]) * dt + sigma[t-1] * dt * z
        # print(f'theta = {np.mean(theta)}, mean = {np.mean(model_rate)}, std = {np.std(model_rate)}')
        model_bond_price = np.exp(-(sum(r[:t-1]) + model_rate)*dt)
        # print(f'mean = {np.mean(model_bond_price)}, std = {np.std(model_bond_price)}')
        # print(f'actual_bond_price = {actual_bond_price}, model_bond_price = {model_bond_price}, diff = {actual_bond_price-model_bond_price}')
        error = np.mean(np.abs(actual_bond_price - model_bond_price))
        # print('error: ', error)
        return error
    theta_optimized = minimize(objective_function, np.array([0.0312]*number_paths), options={'maxiter':10}, tol=1e-6).x
    print(f'theta_optimized = {np.mean(theta_optimized)}')
    theta[t-1] = np.mean(theta_optimized)
    theta[t-1] = market_data[t-1]
    r[t] = r[t-1] + alpha * (theta[t-1] - r[t-1]) * dt + sigma[t-1] * dt * z 
    print(f'theta = {np.mean(theta)}')
    print(f'r = {np.mean(r[t])}')

print(f'mean of rate: {np.mean(r, axis = 1)}')
print(f'theta : {theta}')