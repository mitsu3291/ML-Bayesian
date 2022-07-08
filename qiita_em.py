import os
import numpy as np
import numpy.random as rd
import scipy as sp
from scipy import stats as st
from collections import Counter
import util

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

def calc_likelihood(data, mu, sigma, pi, K):
    N = len(data)
    likelihood = np.zeros((N, K))
    for k in range(K):
        likelihood[:, k] = [pi[k]*st.multivariate_normal.pdf(d, mu[k], sigma[k]) for d in data]
    return likelihood

if __name__ == "__main__":
    # read csv
    file_path = "x.csv"
    x = util.get_data(file_path)

    # initialize pi
    K = 4
    N = len(x)
    dim_x = len(x[0])
    pi = np.zeros(K)
    for k in range(K):
        if k == K-1:
            pi[k] = 1 - np.sum(pi)
        else:
            pi[k] = 1/K

    # initialize mu
    max_x, min_x = np.max(x[:,0]), np.min(x[:,0])
    max_y, min_y = np.max(x[:,1]), np.min(x[:,1])
    max_z, min_z = np.max(x[:,2]), np.min(x[:,2])
    mu = np.c_[np.random.uniform(low=min_x, high=max_x, size=K), np.random.uniform(low=min_y, high=max_y, size=K), np.random.uniform(low=min_z, high=max_z, size=K)]
    sigma = np.array([np.eye(3), np.eye(3), np.eye(3), np.eye(3)])

    for iter in range(100):
        # E step ========================================================================
        # calculate responsibility(負担率)
        likelihood = calc_likelihood(x, mu, sigma, pi, K)
        #gamma = np.apply_along_axis(lambda x: [xx/np.sum(x) for xx in x] , 1, likelihood)
        gamma = (likelihood.T/np.sum(likelihood, axis=1)).T
        N_k = np.array([np.sum(gamma[:,k]) for k in range(K)])

        # M step ========================================================================

        # caluculate pi
        pi =  N_k/N

        # calculate mu
        tmp_mu = np.zeros((K, dim_x))

        for k in range(K):
            for i in range(N):
                tmp_mu[k] += gamma[i, k]*x[i]
            tmp_mu[k] = tmp_mu[k]/N_k[k]
            #print('updated mu[{}]:\n'.format(k) , tmp_mu[k])
        mu_prev = mu.copy()
        mu = tmp_mu.copy()
        #print('updated mu:\n', mu)

        # calculate sigma
        tmp_sigma = np.zeros((K, dim_x, dim_x))

        for k in range(K):
            tmp_sigma[k] = np.zeros((dim_x, dim_x))
            for i in range(len(x)):
                tmp = np.asanyarray(x[i]-mu[k])[:,np.newaxis]
                tmp_sigma[k] += gamma[i, k]*np.dot(tmp, tmp.T)
            tmp_sigma[k] = tmp_sigma[k]/N_k[k]

            #print('updated sigma[{}]:\n'.format(k) , tmp_sigma[k])
        sigma = tmp_sigma.copy()

        # calculate likelihood
        prev_likelihood = likelihood
        likelihood = calc_likelihood(x, mu, sigma, pi, K)

        prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
        sum_log_likelihood = np.sum(np.log(likelihood))
        diff = prev_sum_log_likelihood - sum_log_likelihood

        if np.abs(diff) < 0.0001:
            print("likelihood is converged")
            print(f"simga : {sigma}")
            print(f"mu : {mu}")
        else:
            print(f"iter : {iter}, diff : {diff}")
