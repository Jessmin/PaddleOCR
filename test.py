import numpy as np


# 定义softmax函数
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# 利用numpy计算
def cross_entropy_np(x, y):
    loss = -np.sum(x) / len(y)
    return loss


if __name__ == '__main__':
    # 假设有数据x, y
    x = np.array([1, 2, 3])

    y = np.array([1, 2, 3])
    print('numpy result: ', cross_entropy_np(x, y))
# numpy result:  1.0155949508195155
