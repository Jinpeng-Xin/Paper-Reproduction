#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt  
import matplotlib  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

plt.rcParams['font.family']=' Times New Roman, SimSun'# 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
matplotlib.rcParams['axes.unicode_minus'] = False  
plt.rcParams['figure.dpi'] = 300  # plt.show显示分辨率
# 测试绘图  
plt.figure()  
plt.plot([1, 2, 3], [1, 2, 3])  
plt.title("测试图表 - 中文显示和Times New Roman字体")  
plt.xlabel("X轴 - Times New Roman")  
plt.ylabel("Y轴 - 宋体")  
plt.show() 


# In[3]:


import pandas as pd
import numpy as np
data2=r".\data.xlsx"
df=pd.read_excel(data2)
print(df.describe().T)


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# 计算皮尔逊相关系数矩阵
corr_matrix = df.iloc[:, 1:].corr(method='pearson')

# 使用 seaborn 绘制热图来显示相关性
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Heatmap')
plt.show()


# In[4]:


# 使用 seaborn 绘制 pairplot 来显示数据分布和相关性
sns.pairplot(df)
plt.suptitle('Pairplot of Data Distribution and Correlation', y=1.02)
plt.show()


# In[111]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim


# 划分特征和目标变量
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 将数据转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# 定义 ANN 模型
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 初始化模型
input_size = X_train.shape[1]
model = ANN(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)

# 训练模型
num_epochs = 1000
train_losses = []
r2_scores = []

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 在训练过程中计算 R^2 分数
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train)
        r2 = r2_score(y_train.numpy(), train_preds.numpy())
        r2_scores.append(r2)
    model.train()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, R2 Score: {r2:.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    test_r2 = r2_score(y_test.numpy(), test_outputs.numpy())
    print(f'Test Loss: {test_loss.item():.4f}, Test R2 Score: {test_r2:.4f}')

# 绘制训练损失曲线和 R^2 分数曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss and R2 Score')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[112]:


# 绘制训练集和测试集的回归图
plt.figure(figsize=(12, 6))

# 训练集回归图
plt.subplot(1, 2, 1)
with torch.no_grad():
    train_preds = model(X_train).numpy()
plt.scatter(y_train.numpy(), train_preds)
plt.plot([y_train.numpy().min(), y_train.numpy().max()], [y_train.numpy().min(), y_train.numpy().max()], 'r--')
plt.xlabel('True Values (Training Set)')
plt.ylabel('Predictions (Training Set)')
plt.title(f'Training Set Regression Plot (R2: {r2:.4f})')

# 测试集回归图
plt.subplot(1, 2, 2)
with torch.no_grad():
    test_preds = model(X_test).numpy()
plt.scatter(y_test.numpy(), test_preds)
plt.plot([y_test.numpy().min(), y_test.numpy().max()], [y_test.numpy().min(), y_test.numpy().max()], 'r--')
plt.xlabel('True Values (Test Set)')
plt.ylabel('Predictions (Test Set)')
plt.title(f'Test Set Regression Plot (R2: {test_r2:.4f})')

plt.tight_layout()
plt.show()


# In[113]:


# 绘制测试集上预测值与真实值的误差图
errors = test_preds - y_test.numpy()
indices = range(len(errors))

plt.figure(figsize=(12, 6))
plt.plot(indices, errors, label='Prediction Errors')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error Line')
plt.xlabel('Sample Index')
plt.ylabel('Prediction Error')
plt.title('Test Set Prediction Error Plot')
plt.legend()
plt.show()


# In[114]:


# 构建误差消除框架

# 计算训练集的预测误差
with torch.no_grad():
    train_preds = model(X_train)
    test_preds = model(X_test)
    train_errors = train_preds - y_train


# In[118]:


# 定义 ANN 模型
class ErrorCorrectionModel(nn.Module):
    def __init__(self, input_size):
        super(ErrorCorrectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 初始化模型
input_size = X_train.shape[1]
model_error = ErrorCorrectionModel(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model_error.parameters(), lr=0.003)

# 训练模型
num_epochs = 2000
train_losses = []
r2_scores = []

for epoch in range(num_epochs):
    # 前向传播
    outputs = model_error(X_train)
    loss = criterion(outputs, train_errors)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 在训练过程中计算 R^2 分数
    model.eval()
    with torch.no_grad():
        train_error_preds = model_error(X_train)
        r2 = r2_score(train_errors.numpy(), train_error_preds.numpy())
        r2_scores.append(r2)
    model_error.train()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, R2 Score: {r2:.4f}')

# 评估模型
model_error.eval()
with torch.no_grad():
    test_error_outputs = model_error(X_test)
    test_loss = criterion(test_error_outputs, test_preds - y_test)
    test_r21 = r2_score(test_preds - y_test, test_error_outputs.numpy())
    print(f'Test Loss: {test_loss.item():.4f}, Test R2 Score: {test_r21:.4f}')

# 绘制训练损失曲线和 R^2 分数曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss and R2 Score')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[120]:


# 绘制误差消除前后的测试集回归图
plt.figure(figsize=(12, 6))
R2 = r2_score(y_test, test_preds-test_error_outputs.numpy())
# 误差消除前
plt.subplot(1, 2, 1)
plt.scatter(y_test.numpy(), test_preds.numpy())
plt.plot([y_test.numpy().min(), y_test.numpy().max()], [y_test.numpy().min(), y_test.numpy().max()], 'r--')
plt.xlabel('True Values (Test Set)')
plt.ylabel('Predictions (Test Set)')
plt.title(f'Test Set Regression Plot Before Correction (R2: {test_r2:.4f})')

# 误差消除后
plt.subplot(1, 2, 2)
plt.scatter(y_test.numpy(), test_preds-test_error_outputs)
plt.plot([y_test.numpy().min(), y_test.numpy().max()], [y_test.numpy().min(), y_test.numpy().max()], 'r--')
plt.xlabel('True Values (Test Set)')
plt.ylabel('Predictions (Test Set)')
plt.title(f'Test Set Regression Plot After Correction (R2: {R2:.4f})')

plt.tight_layout()
plt.show()


# In[121]:


# 绘制测试集上预测值与真实值的误差图
indices = range(len(errors))

plt.figure(figsize=(12, 6))
plt.plot(indices,y_test, label='Truth')
plt.plot(indices, test_preds, label='Prediction Before Errors')
plt.plot(indices, test_preds-test_error_outputs, label='Prediction After Errors')
plt.xlabel('Sample Index')
plt.ylabel('Prediction Error')
plt.title('Test Set Plot')
plt.legend()
plt.show()


# In[ ]:




