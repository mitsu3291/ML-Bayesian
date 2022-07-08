import matplotlib.pyplot as plt
import util
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_zlabel("z", size = 14)

    x = util.get_data(file_path="x.csv")
    x = util.return_xyz(x)

    ax.scatter(x[0], x[1], x[2])
    plt.show()