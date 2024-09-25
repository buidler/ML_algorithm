# -*- coding:utf-8 -*-
# @Time: 2024/9/24 
# @Author:ZhangJiaYuan
# @File:Uni variateLinearRegression.py
# @Software: PyCharm

"""单变量线性回归"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('../data/Folds5x2_pp.csv')
# 得到训练和测试数据
train_data = data.sample(frac=0.9)
# sample 从所选的数据的指定 axis 上返回随机抽样结果，类似于random.sample()函数。
test_data = data.drop(train_data.index)
# drop 删除数据集中多余的数据

input_param_name = 'AT'
output_param_name = 'PE'

x_train = train_data[[input_param_name]].values
# .values为了变成ndarray模式
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
# panda中，单括号返回一维数组，双括号返回一个二维数组，即使只有一列
y_test = test_data[[output_param_name]].values

# plt.scatter(x_train, y_train, label='Train data')
# plt.scatter(x_test, y_test, label='test data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.title('above us only sky')
# plt.legend()
# plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时候的损失', cost_history[0])
print('训练后的损失', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('Cost')
plt.show()

predictions_num = 3000
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('above us only sky')
plt.legend()
plt.show()
