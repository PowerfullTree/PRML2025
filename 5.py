import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 读取 Excel 文件
df = pd.read_excel('D:\在北航\\2025春课程资料\模式识别与机器学习-秦曾昌\第一次作业\data.xlsx', sheet_name='Sheet1')

# 提取四组数据
X_train = df['x'].values.reshape(-1, 1)  # 将X_train转换为列向量
y_train = df['y_complex'].values.reshape(-1, 1)
X_test = df['x_new'].values.reshape(-1, 1)
y_test = df['y_new_complex'].values.reshape(-1, 1)

# 设置多项式的阶数
degree = 5  # 可以调整这个值来选择不同阶数的多项式

# 创建多项式特征
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 使用线性回归进行拟合
poly_regressor = LinearRegression()
poly_regressor.fit(X_train_poly, y_train)

# 预测
y_train_pred_poly = poly_regressor.predict(X_train_poly)
y_test_pred_poly = poly_regressor.predict(X_test_poly)

# 计算误差
train_error_poly = mean_squared_error(y_train, y_train_pred_poly)
test_error_poly = mean_squared_error(y_test, y_test_pred_poly)

# 输出训练误差和测试误差
print("多项式拟合 - 训练误差:", train_error_poly)
print("多项式拟合 - 测试误差:", test_error_poly)

# 绘制拟合结果
plt.figure(figsize=(10, 6))

# 绘制训练数据
plt.scatter(X_train, y_train, color='blue', label='train data')
plt.scatter(X_test, y_test, color='green', label='test data')

# 绘制多项式拟合曲线
X_all = np.linspace(0, 10, 1000).reshape(-1, 1)  # 生成更细的X用于绘制拟合曲线
X_all_poly = poly.transform(X_all)
y_all_pred_poly = poly_regressor.predict(X_all_poly)
plt.plot(X_all, y_all_pred_poly, color='red', label=f'Polinomial Feather (degree={degree})')

# 标题和标签
plt.title('Polinomial Feather')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 显示图形
plt.show()
