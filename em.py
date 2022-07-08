from estimater import EM
import util
import numpy as np

if __name__ == "__main__":
    # Params
    file_path = "x.csv"
    x = util.get_data(file_path)
    N = len(x)
    K = 4
    
    pi = np.array([0.25]*4)
    mu = np.arange(12).reshape(4,3)/10
    sigma = np.array([np.eye(3), np.eye(3), np.eye(3), np.eye(3)])

    em = EM(x, N, K, pi, mu, sigma)
    em.solve()
