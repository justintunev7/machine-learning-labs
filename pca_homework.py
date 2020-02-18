import numpy as np

cov = np.array([[0.715, -1.39],[-1.39, 2.72]])

print("COVARIANCE:\n",cov)
print("\nEIGENVALUES AND VECTORS\n", np.linalg.eig(cov))

p = np.array([0.45554483, -0.89021285])
data = np.array([[0.2, -0.2],[-1.1, 2.1],[1, -2.1],[0.5, -0.9],[-0.6, 1.1]])

print("\nNEW DATA WITH SINGLE FEATURE:\n",np.matmul(data, p))
print(np.dot(data,p))