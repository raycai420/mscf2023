import numpy as np
from scipy.optimize import minimize

T = 10  # Total time period
num_periods = 10
dt = T / num_periods
number_paths = 10000
market_data = np.array([0.0312, 0.0320, 0.0325, 0.0328, 0.0333, 0.0337, 0.0340, 0.0343, 0.02345, 0.0347])
sigma = np.array([0.15, 0.15, 0.15, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
alpha = 0.1

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
        print(actual_bond_price)
        model_rate = r[t-1] + alpha * (theta - r[t-1]) * dt + sigma[t-1] * dt * z
        print(f'shape = {np.shape(actual_bond_price)}')
        model_bond_price = np.mean(np.exp(-(r[t-1] + model_rate)*(t+1)))
        error = np.abs(actual_bond_price - model_bond_price)
        return error
    theta[t-1] = minimize(objective_function, 0.03, method='BFGS').x
    r[t] = r[t-1] + alpha * (theta[t-1] - r[t-1]) * dt + sigma[t-1] * dt * z 

print(np.mean(r, axis = 1))
