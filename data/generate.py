# 生成仿真实验环境数据
import pandas as pd
import random
import os

root_path = os.getcwd()
data_path = root_path + "/data/"
bandwidth_file_name = "bandwidth.csv"


def generate_bandwidth_data():
    # 生成仿真带宽数据
    time_range = 120
    time_data = list(range(1, time_range + 1))
    bandwidth_data = [10]
    for _ in range(time_range - 1):
        pre = bandwidth_data[-1]
        pre += random.random() * 2 - 1
        if pre < 5:
            pre = 5 + random.random()
        if pre > 15:
            pre = 15 + random.random()
        bandwidth_data.append(pre)
    data = pd.DataFrame({'time': time_data, 'bandwidth': bandwidth_data})
    return data


def save_data(data, file_name):
    data.to_csv(data_path + file_name, index=False, sep=',')


def read_data(file_name):
    data = pd.read_csv(data_path + file_name, index=False, sep=',')
    return data


def read_bandwidth_data():
    return read_data(bandwidth_file_name)


def main():
    # bandwidth_data = generate_bandwidth_data()
    # save_data(bandwidth_data, bandwidth_file_name)
    return


if __name__ == "__main__":
    main()
