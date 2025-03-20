import matplotlib.pyplot as plt



def f1():
    # 数据
    loss = [1805.794434, 1731.413208, 1698.965088, 1659.385376, 1624.559326, 1594.250610, 1568.514893, 1546.414185, 1527.241577,
            1510.437378]
    idx = [1,2,3,4,5,6,7,8,9,10]
    train = [66.51, 66.53, 66.53, 66.51, 66.49, 66.44, 66.38, 66.29, 66.20, 66.08]
    test = [66.95, 66.95, 66.95, 80.14, 80.01, 79.98, 79.94, 79.94, 79.94, 79.94]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 创建折线图
    ax1.plot(idx, train, marker='o', linestyle='-', color='b', label='train')
    ax1.plot(idx, test, marker='o', linestyle='-', color='r', label='test')
    ax2.plot(idx, loss, marker='o', linestyle='-', color='g', label='loss')

    # 添加标题和标签
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")

    # 添加网格
    ax1.grid(True)
    ax2.grid(True)
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


def f2():
    # 数据
    loss = [2001.338379, 1975.564331, 1966.852295, 1963.425049, 1961.673706, 1960.303833, 1960.058838, 1959.023315, 1957.695679, 1956.727417]
    idx = [1,2,3,4,5,6,7,8,9,10]
    train = [66.51, 66.54, 66.54, 66.54, 66.54, 66.54, 66.54, 66.54, 66.54, 66.54]
    test = [66.95, 66.95, 66.95, 66.95, 66.95, 66.95, 66.95, 66.95, 66.95, 66.95]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 创建折线图
    ax1.plot(idx, train, marker='o', linestyle='-', color='b', label='train')
    ax1.plot(idx, test, marker='o', linestyle='-', color='r', label='test')
    ax2.plot(idx, loss, marker='o', linestyle='-', color='g', label='loss')

    # 添加标题和标签
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")

    # 添加网格
    ax1.grid(True)
    ax2.grid(True)
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

if __name__ == '__main__':
    f2()
