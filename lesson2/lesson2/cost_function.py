from dataset import get_beans
import numpy as np
import matplotlib.pyplot as plt
import sys


DATASET_SIZE = 100
xs, ys = get_beans(DATASET_SIZE)

def get_avg_es(w):
    y_predict = w * xs
    error_squared = (ys - y_predict) ** 2 # 平方误差
    # 注：平方误差和方差不是一个东西，平方误差是单个值与其预测值之差的平方，方差是每个值与均值之差的平方之和的平均数
    # 均方误差是所有平方误差之和再求平均
    avg_es = (1/DATASET_SIZE)*np.sum(error_squared) # 均方误差
    return avg_es

plt.title("Cost Function")
plt.xlabel('Weight')
plt.ylabel('Mean Squared Error')

ws = np.arange(0, 3, 0.1)
es = []
for w in ws:
    avg_es = get_avg_es(w)
    es.append(avg_es)

# plt.scatter(xs, ys) # original dataset scatter plot

plt.plot(ws, es) # cost function

plt.show()

w_best = np.sum(xs * ys) / np.sum(xs * xs) # normal equation
print("minimum w value (derivative 0):", w_best)

best_prediction = w_best * xs
plt.title("Best prediction")
plt.xlabel('Size')
plt.ylabel('Toxicity')

plt.scatter(xs, ys)
plt.plot(xs, best_prediction)
plt.show()