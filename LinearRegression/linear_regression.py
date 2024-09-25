import numpy as np

from utils.features.prepare_for_training import prepare_for_training

class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据预处理
        2.先得到所有的特征个数
        3.初始化参数矩阵
        :param data: 样本数据
        :param labels: 真实值
        :param polynomial_degree:
        :param sinusoid_degree:
        :param normalize_data:
        :return:
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.feature_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        num_features = self.data.shape[1]
        # 有几个特征，就有几个theta
        self.theta = np.zeros((num_features, 1))
        # 生成一个num_features行1列的矩阵

    def train(self, alpha, num_iterations=500):
        """
        线性回归训练
        :param alpha: 学习率
        :param num_iterations: 迭代器
        :return: 参数和损失元组
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        小批量梯度下降算法
        :param alpha: 步长
        :param num_iterations: 迭代器
        :return: 损失列表
        """
        cost_history = []
        for i in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        小批量梯度下降参数更新计算
        :param alpha:步长
        :return:
        """
        num_examples = self.data.shape[0]
        # 一次下降的样本数
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        # 预测减去真实即为误差
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        # 用矩阵乘法不用循环,T是转秩
        self.theta = theta

    def cost_function(self, data, labels):
        """
        计算损失函数
        :param data: 样本数据
        :param labels: 真实值
        :return: 损失值
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta)/num_examples
        # 平方误差的矩阵表示
        return cost[0][0]
        # print(cost.shape)
        # print(cost)
        # 当前这一步损失值的位置就在00，找不到可以print找

    @staticmethod
    def hypothesis(data, theta):
        """
        计算预测值
        :param data: 样本数据
        :param theta: 参数
        :return: 预测值
        """
        predictions = np.dot(data, theta)
        # dot函数可以用于计算两个向量的点积, 也可以用于矩阵乘法。
        return predictions

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data, self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用线性回归预测值
        :param data:
        :return: 预测值
        """
        data_processed = prepare_for_training(data, self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions
