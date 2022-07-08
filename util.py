import numpy as np

def get_data(file_path):
    file = open(file_path)
    data_list = []
    for data in file:
        data = data.split(',')
        data_list.append([float(data[i]) for i in range(len(data))])

    return np.array(data_list)

def return_xyz(data_list):
    x_list = []
    y_list = []
    z_list = []
    for data in data_list:
        x_list.append(data[0])
        y_list.append(data[1])
        z_list.append(data[2])
    return_list = [x_list, y_list, z_list]

    return return_list

if __name__ == "__main__":
    file_path = "x.csv"
    x = get_data(file_path)
    print(x)
    print(len(x))