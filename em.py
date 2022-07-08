from estimater import EM
import util
import numpy as np

if __name__ == "__main__":
    # Params
    file_path = "x.csv"
    x = util.get_data(file_path)
    N = len(x)
    K = 2
    max_x, min_x = np.max(x[:,0]), np.min(x[:,0])
    max_y, min_y = np.max(x[:,1]), np.min(x[:,1])
    max_z, min_z = np.max(x[:,2]), np.min(x[:,2])
    mu = np.c_[np.random.uniform(low=min_x, high=max_x, size=K), np.random.uniform(low=min_y, high=max_y, size=K), np.random.uniform(low=min_z, high=max_z, size=K) ]
    print(f"mu : {mu}")
    pi = np.array([0.3, 0.3, 0.2, 0.2])
    mu = np.array([[3, -2, 3],
                   [-3, 2, -3],
                   [-5, 0, 5],
                   [5, 0, -5]])
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
    sigma = np.array([np.eye(3), np.eye(3), np.eye(3), np.eye(3)])

    em = EM(x, N, K, pi, mu, sigma)
    em.solve()
