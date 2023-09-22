import numpy as np

r = np.array([0.0312, 0.0320, 0.0325, 0.0328, 0.0333, 0.0337, 0.0340, 0.0343, 0.0345, 0.0347])
sigma = np.array([0, 0.15, 0.15, 0.15, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
t = np.array([1,2,3,4,5,6,7,8,9,10])
N = 10
alpha = 0.1

d = np.exp(-r*t)

theta = np.zeros(N)
mu_r = np.zeros(N)
sigma_r = np.zeros(N)

mu_r[0] = r[0]
sigma_r[0] = 0
for i in range(1, N):

    
    ri_term = np.exp(-(mu_r[i-1]*(1-alpha)) + 0.5*(sigma_r[i-1]*(1-alpha))**2)

    z_term = np.exp(0.5*sigma[i]**2)
    theta[i] = np.log(d[i]/(d[i-1]*ri_term*z_term))/(-alpha)
    
    mu_r[i] = mu_r[i-1]*(1-alpha) + alpha*theta[i]
    sigma_r[i] = np.sqrt((sigma_r[i-1]*(1-alpha))**2 + sigma[i]**2)

print("Theta:", list(theta))
print("E(R_(t, i+1):", list(mu_r))
print("V(R_(t, t+1):", list(sigma_r))

Eert = np.exp(-mu_r+0.5*np.square(sigma_r))
print("E(Product of expectations of e^-r):", [np.prod(Eert[:i]) for i in range(1, N+1)])
print("Discount Factors (should be equal to above):", list(d))