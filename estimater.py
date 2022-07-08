import numpy as np
from scipy.stats import multivariate_normal

class EM:
    def __init__(self, x, N, K, pi, mu, sigma):
        self.x = x
        self.dim_x = len(x[0])
        self.N = N
        self.K = K
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.likelihood = None
        self.gamma = None
        self.eps = 1e-4

    def solve(self):
        for iter in range(100):
            self.__calc_estep()
            self.__calc_mstep()
            prev_likelihood = self.likelihood
            likelihood = self.__calc_likelihood()
            prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
            sum_log_likelihood = np.sum(np.log(likelihood))
            diff = prev_sum_log_likelihood - sum_log_likelihood
            if np.abs(diff) < self.eps:
                print("likelihood is converged")
            else:
                print(f"iter : {iter}, diff : {np.abs(diff)}")

    def __calc_likelihood(self):
        likelihood = np.zeros((self.N, self.K))
        for k in range(self.K):
            likelihood[:, k] = [self.pi[k]*multivariate_normal.pdf(xn, self.mu[k], self.sigma[k]) for xn in self.x]

        return likelihood

    def __calc_estep(self):
        self.likelihood = self.__calc_likelihood()
        self.gamma = (self.likelihood.T/np.sum(self.likelihood, axis=1)).T

    def __calc_mstep(self):
        S_k1 = np.array([np.sum(self.gamma[:,k]) for k in range(self.K)])
        S_kx = np.array([np.sum((self.gamma[:,k]*self.x.T).T, axis=0) for k in range(self.K)])
        xx = np.zeros((self.N, self.dim_x, self.dim_x))
        for n in range(self.N):
            x_k = self.x[n].reshape(self.dim_x,1)
            xx[n] = x_k@x_k.T
        S_kxx = np.array([np.sum((self.gamma[:,k]*xx.T).T, axis=0) for k in range(self.K)])

        self.pi = S_k1/self.N
        self.mu = (S_kx.T/S_k1).T
        self.sigma = np.zeros((self.K, self.dim_x, self.dim_x))
        for k in range(self.K):
            self.sigma[k] = S_kxx[k]/S_k1[k] - self.mu[k].reshape(self.dim_x,1)@self.mu[k].reshape(self.dim_x,1).T