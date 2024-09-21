import matplotlib.pyplot as plt  


# 年份数据
year = [
    2004, 2005, 2006, 2007, 2008, 2009, 2010, 
    2011, 2012, 2013, 2014, 2015, 2016, 2017, 
    2018, 2019, 2020, 2021, 2022
]

# 房价数据，
prices = [
    4355.94, 5041.21, 6152.17, 8439.06, 8781, 8988, 10615,
    10925.84, 12000.88, 13954, 14739, 14083, 16346, 17685,
    21581.78, 24015, 27112, 30580, 29455
]


# 数据标准化（归一化处理）函数
# 将数据缩放到 [0, 1] 区间，方便梯度下降算法更快收敛
def normalize(data):
    min_val = min(data)  # 找到数据中的最小值
    max_val = max(data)  # 找到数据中的最大值
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]  # 归一化公式
    return normalized_data  # 返回归一化后的数据

# 调用归一化函数，对年份数据进行标准化
normalized_year = normalize(year)

# 初始化回归模型的参数，斜率（m）和截距（b）
m = 0  # 初始斜率设为0
b = 0  # 初始截距设为0

# 设置梯度下降算法的超参数
learning_rate = 0.01  # 学习率
epochs = 10000  # 训练迭代的次数
n = len(year)  # 数据点的数量

# 开始梯度下降算法
for epoch in range(epochs):
    # 根据当前的m和b，计算所有年份的预测房价
    y_pred = [m * x + b for x in normalized_year]  # 预测值 y = m*x + b

    # 计算损失（均方误差），衡量预测值与真实值之间的差异
    loss = sum((y_p - y) ** 2 for y_p, y in zip(y_pred, prices)) / n  

    # 计算损失函数对m的偏导数（梯度）
    dm = (2/n) * sum((y_p - y) * x for y_p, y, x in zip(y_pred, prices, normalized_year))

    # 计算损失函数对b的偏导数（梯度）
    db = (2/n) * sum((y_p - y) for y_p, y in zip(y_pred, prices))

    # 根据梯度更新参数m和b
    m -= learning_rate * dm  # 更新斜率 m
    b -= learning_rate * db  # 更新截距 b

    if (epoch + 1) % 1000 == 0:
        print(f"迭代次数: {epoch + 1}, 损失: {loss:.4f}, m: {m:.6f}, b: {b:.6f}")

# 数据标准化的最小值和最大值
year_min = min(year)  
year_max = max(year)  # 年份中的最大值

# 将归一化后的斜率和截距转换回原始尺度
m_original = m / (year_max - year_min)  # 原始斜率
b_original = b - (m_original * year_min)  # 原始截距

# 打印最终的回归直线方程
print(f"\n回归直线方程: y = {m_original:.2f}x + {b_original:.2f}")

# 使用最终的m和b，预测所有年份的房价
predicted_prices = [m_original * x + b_original for x in year]


# 开始绘图
plt.figure(figsize=(12, 6))
# 绘制实际的房价数据
plt.scatter(year, prices, color='blue', label='Real Price')
# 绘制回归直线
plt.plot(year, predicted_prices, color='red', label='Regression line')


plt.title('')
plt.xlabel('Years')  # x轴
plt.ylabel('House Price')  # y轴
plt.xticks(year)  # 将x轴的刻度设置为每个年份
plt.legend()
plt.grid(True)
plt.show()






