import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. 加载数据并预处理
df = pd.read_csv('D:/在北航/2025春课程资料/模式识别与机器学习-秦曾昌/第3次作业/LSTM-Multivariate_pollution1.csv')
df = df.sort_index()  # 确保时间顺序

# 处理分类变量（风向）
df = pd.get_dummies(df, columns=['wnd_dir'])
wind_dir_cols = [col for col in df.columns if 'wnd_dir_' in col]
df[wind_dir_cols] = df[wind_dir_cols].astype(np.float32)

# 统一所有特征为float32
df = df.astype('float32')

# 2. 创建监督学习数据集
target = df['pollution'].shift(-1)  # 下一个小时的污染值
features = df.copy()

# 删除最后一行（包含NaN）
features = features.iloc[:-1]
target = target.dropna()

# 3. 数据标准化
num_cols = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
feature_scaler = MinMaxScaler()
features[num_cols] = feature_scaler.fit_transform(features[num_cols])

target_scaler = MinMaxScaler()
target = target_scaler.fit_transform(target.values.reshape(-1, 1))

# 4. 转换为LSTM输入格式
X = features.values.reshape((features.shape[0], 1, features.shape[1]))
y = target

# 5. 划分训练集/测试集
train_size = int(len(X) * 0.9921)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. 创建LSTM模型
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# 7. 训练模型
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    verbose=2)

# ======== 新增：绘制损失函数曲线 ========
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 8. 预测测试集
y_pred = model.predict(X_test)

# 反标准化结果
y_pred_inv = target_scaler.inverse_transform(y_pred)
y_test_inv = target_scaler.inverse_transform(y_test)

# 9. 评估结果
rmse = np.sqrt(np.mean((y_pred_inv - y_test_inv) ** 2))
print(f"Test RMSE: {rmse:.2f}")

# 10. 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Pollution Prediction Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Pollution Level')
plt.legend()
plt.show()