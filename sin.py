import numpy as np
import matplotlib.pyplot as plt

def fit(x_data: np.ndarray, y_data: np.ndarray, learning_rate: float, num_iterations: int) -> np.ndarray:
    coefficients = np.random.rand(6)  # 随机初始化6个系数

    for iteration in range(1, num_iterations + 1):
        # 预测值计算
        predicted_y = (coefficients[0] + coefficients[1] * x_data + 
                       coefficients[2] * x_data ** 2 + 
                       coefficients[3] * x_data ** 3 + 
                       coefficients[4] * x_data ** 4 + 
                       coefficients[5] * x_data ** 5)

        #
        if iteration % 1000 == 0:
            print(f"Iteration: {iteration}, Loss: {calculate_loss(predicted_y, y_data):.4f}")

        # 计算梯度
        gradients = np.zeros(6)
        for i in range(6):
            gradients[i] = np.mean(2 * (predicted_y - y_data) * (x_data ** i))

        # 更新系数
        coefficients -= learning_rate * gradients

    return coefficients

#写个计算loss的函数
def calculate_loss(predicted_y: np.ndarray, y_data: np.ndarray) -> float:
    return np.mean((predicted_y - y_data) ** 2)

# 生成训练数据
x_data = np.linspace(-np.pi, np.pi, 100)
y_data = np.sin(x_data)

# 设置学习率和迭代次数
learning_rate = 0.0001
num_iterations = 50000 #发现这个迭代得越大，loss越小，拟合效果更好一点

# 训练模型
final_coeffs = fit(x_data, y_data, learning_rate, num_iterations)
predicted_y = (final_coeffs[0] + final_coeffs[1] * x_data + 
               final_coeffs[2] * x_data ** 2 + 
               final_coeffs[3] * x_data ** 3 + 
               final_coeffs[4] * x_data ** 4 + 
               final_coeffs[5] * x_data ** 5)

#画个图
plt.figure(figsize=(10, 6))
plt.title('')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_data, y_data, color='red', label='sin(x)')
plt.plot(x_data, predicted_y, linestyle='--', color='blue', label='Predicted')
plt.legend()
plt.show()
