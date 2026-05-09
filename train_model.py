import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 加载数据
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]  # 特征 (工作经验)
y = data.iloc[:, -1]   # 目标 (薪水)

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 保存模型文件 (二进制格式)
joblib.dump(model, 'linear_model.pkl')

# 保存模型参数到文本文件 (方便人类阅读)
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')