import numpy as np
from scipy.stats import multivariate_normal
import util

def calc_likelihood(data, mu, sigma, pi, N, class_num):
    likelihood = np.zeros((N, class_num)) # N=1000, class_num=4
    for k in range(class_num):
        likelihood[:, k] = [pi[k]*multivariate_normal.pdf(d, mu[k], sigma[k]) for d in data]

    return likelihood

pi = np.array([0.3, 0.3, 0.2, 0.2])
mu = np.array([[3, -2, 3],
               [-3, 2, -3],
               [-5, 0, 5],
               [5, 0, -5]])

file_path = "x.csv"
x = util.get_data(file_path)
x = np.array(x)

sigma = np.array([[[1.5, -1, 1], 
                   [-1, 2, -1],
                   [1, -1, 1]],
                  [[1.5, -1, 1], 
                   [-1, 2, -1],
                   [1, -1, 1]],
                  [[1.5, -1, 1], 
                   [-1, 2, -1],
                   [1, -1, 1]],
                  [[1.5, -1, 1], 
                   [-1, 2, -1],
                   [1, -1, 1]]])

# params
N = 10000
class_num = 4

# E step
likelihood = calc_likelihood(data=x, mu=mu, sigma=sigma, pi=pi, N=N, class_num=class_num)

gamma = (likelihood.T/np.sum(likelihood, axis=1)).T
S_k1 = np.array([np.sum(gamma[:,k]) for k in range(4)])
S_kx = np.array([np.sum((gamma[:,k]*x.T).T, axis=0) for k in range(class_num)])
xx = np.zeros((N, len(x[0]), len(x[0])))
for k in range(class_num):
    x_k = x[k].reshape(len(x[0]),1)
    xx[k] = x_k@x_k.T
S_kxx = np.array([np.sum((gamma[:,k]*xx.T).T, axis=0) for k in range(class_num)])

# M step
pi = S_k1/N
mu_prev = mu.copy()
mu = (S_kx.T/S_k1).T

sigma = np.zeros((class_num, len(x[0]), len(x[0])))
for k in range(class_num):
    sigma[k] = S_kxx[0]/S_k1[k] - mu_prev[k].reshape(3,1)@mu_prev[k].reshape(3,1).T

print(sigma)