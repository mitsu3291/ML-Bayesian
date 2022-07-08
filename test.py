import numpy as np

if __name__ == "__main__":
    gamma = np.array([[0,1,2],
                      [0,1,2],
                      [0,1,2]])
    x = np.array([[0,0,0],
                  [1,1,1],
                  [2,2,2]])

    xx = np.zeros((3,3,3))
    for k in range(3):
        x_k = x[k].reshape(3,1)
        xx[k] = x_k@x_k.T

    print(xx)
    print(gamma[:,0])
    print(np.sum(gamma[:,0]*xx, axis=0))
    print(np.sum(gamma[:,1]*xx, axis=0))
    print(np.sum(gamma[:,2]*xx, axis=0))
    #print((gamma[0]*xx.T).T)
    #print([np.sum((gamma[k]*xx.T).T, axis=0) for k in range(3)])

    print(np.array([np.sum((gamma[:,k]*xx.T).T, axis=0) for k in range(3)]))

    z = np.zeros((1,2,3))
    print(z.shape)
    print(z[0])