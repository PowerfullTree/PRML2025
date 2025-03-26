import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 读取 Excel 文件
df = pd.read_excel('D:\在北航\\2025春课程资料\模式识别与机器学习-秦曾昌\第一次作业\data.xlsx', sheet_name='Sheet1')

# 提取四组数据
X_train = df['x'].values.reshape(-1, 1)  # 将X_train转换为列向量
y_train = df['y_complex'].values.reshape(-1, 1)
X_test = df['x_new'].values.reshape(-1, 1)
y_test = df['y_new_complex'].values.reshape(-1, 1)

# 在X前加一列1，以包含偏置项（即截距）
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

#print(X_train_bias)


# 1. 最小二乘法（解析解）
def least_squares(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# 2. 梯度下降法（GD）
def gradient_descent(X, y, lr=0.1, epochs=1000):
    m = X.shape[1]
    theta = np.random.randn(m, 1)  # 初始化theta为随机值
    for _ in range(epochs):
        gradients = 2 / len(X) * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
    return theta

# 3. 牛顿法（Newton's Method）
def newton_method(X, y, epsilon=1e-5, max_iter=1000):
    m = X.shape[1]
    theta = np.random.randn(m, 1)
    for _ in range(max_iter):
        gradient = 2 / len(X) * X.T.dot(X.dot(theta) - y)
        hessian = 2 / len(X) * X.T.dot(X)
        hessian_inv = np.linalg.inv(hessian)
        theta -= hessian_inv.dot(gradient)
        if np.linalg.norm(gradient) < epsilon:
            break
    return theta

# 使用最小二乘法拟合
theta_ls = least_squares(X_train_bias, y_train)

# 使用梯度下降法拟合
theta_gd = gradient_descent(X_train_bias, y_train, lr=0.1, epochs=1000)

# 使用牛顿法拟合
theta_newton = newton_method(X_train_bias, y_train)
theta_gd=theta_newton
# 预测结果
y_pred_ls = X_test_bias.dot(theta_ls)
y_pred_gd = X_test_bias.dot(theta_gd)
y_pred_newton = X_test_bias.dot(theta_newton)



# 计算误差
train_error_ls = mean_squared_error(y_train, X_train_bias.dot(theta_ls))
test_error_ls = mean_squared_error(y_test, y_pred_ls)

train_error_gd = mean_squared_error(y_train, X_train_bias.dot(theta_gd))
test_error_gd = mean_squared_error(y_test, y_pred_gd)

train_error_newton = mean_squared_error(y_train, X_train_bias.dot(theta_newton))
test_error_newton = mean_squared_error(y_test, y_pred_newton)

# 输出误差
print(f"Least Squares - Training Error: {train_error_ls:.4f}, Testing Error: {test_error_ls:.4f}")
print(f"Gradient Descent - Training Error: {train_error_gd:.4f}, Testing Error: {test_error_gd:.4f}")
print(f"Newton's Method - Training Error: {train_error_newton:.4f}, Testing Error: {test_error_newton:.4f}")

# 绘图显示拟合效果
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_test, y_pred_ls, color='red', label='Least Squares Fit')
plt.plot(X_test, y_pred_gd, color='green', label='Gradient Descent Fit')
plt.plot(X_test, y_pred_newton, color='orange', label='Newton\'s Method Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
