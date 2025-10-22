# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:42:12 2025

@author: lenovo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import rasterio as rio
import dask
import matplotlib
import pandas as pd
from shapely.geometry import Point

torch.manual_seed(42)
np.random.seed(42)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从 CSV 文件中读取数据
data = pd.read_csv("F:/zhandianshujvzhengli/df_y1n3.csv")

# 定义特征和目标变量
predictors = (
    ["t2m_amp", "t2m_10y_avg", "ws_10y_avg", "sp_10y_avg", "t2m_mon_1", "ws_mon_1", "sp_mon_1", "t2m_mon_0", "ws_mon_0", "sp_mon_0"]
    + ["ws_" + str(i) for i in range(3)]
    + ["sp_" + str(i) for i in range(3)]
    + ["t2m_" + str(i) for i in range(3)]
    + ["dem", "month", "latitude"]
)
X = data[predictors].values  # 输入特征
y = data["tem"].values.reshape(-1, 1)  # 目标变量

df_test = pd.DataFrame()
df_train = pd.DataFrame()

# 按站点名分组
grouped = data.groupby('name')

# 遍历每个站点
for name, group in grouped:
    # 随机打乱当前站点的数据
    shuffled_group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算验证集的大小（14%）
    validation_size = int(len(shuffled_group) * 0.14)
    
    # 分割验证集和训练集
    validation_group = shuffled_group.iloc[:validation_size]
    train_group = shuffled_group.iloc[validation_size:]
    
    # 将当前站点的验证集和训练集合并到总集中
    df_test = pd.concat([df_test, validation_group])
    df_train = pd.concat([df_train, train_group])

# 重置索引
df_test.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)

# 准备特征和目标变量
X_train = df_train[predictors].values
y_train = df_train["tem"].values.reshape(-1, 1)
X_test = df_test[predictors].values
y_test = df_test["tem"].values.reshape(-1, 1)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)


# 定义模型
class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, depth=3):
        super(SoftDecisionTree, self).__init__()
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaf = 2 ** depth

        self.decision_nodes = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(self.n_internal)])
        self.leaf_values = nn.Parameter(torch.randn(self.n_leaf, 1))

    def forward(self, x):
        batch_size = x.size(0)
        decision_probs = []
        for node in self.decision_nodes:
            p = torch.sigmoid(node(x))
            decision_probs.append(p)
        decision_probs = torch.cat(decision_probs, dim=1)

        leaf_probs = []
        for leaf in range(self.n_leaf):
            path = []
            index = leaf
            for _ in range(self.depth):
                path.append(index % 2)
                index //= 2
            path = path[::-1]
            prob = torch.ones(batch_size, 1).to(device)
            node_index = 0
            for decision in path:
                p = decision_probs[:, node_index].unsqueeze(1)
                if decision == 0:
                    prob = prob * (1 - p)
                else:
                    prob = prob * p
                node_index += 1
            leaf_probs.append(prob)
        leaf_probs = torch.cat(leaf_probs, dim=1)
        output = torch.matmul(leaf_probs, self.leaf_values)
        return output

class SoftRandomForest(nn.Module):
    def __init__(self, n_trees=10, tree_depth=3, input_dim=None):
        super(SoftRandomForest, self).__init__()
        self.n_trees = n_trees
        self.trees = nn.ModuleList([SoftDecisionTree(input_dim, depth=tree_depth) for _ in range(n_trees)])

    def forward(self, x):
        preds = [tree(x) for tree in self.trees]
        preds = torch.stack(preds, dim=0)
        return torch.mean(preds, dim=0)

class SoftXGBoost(nn.Module):
    def __init__(self, n_estimators=10, tree_depth=3, learning_rate=0.1, input_dim=None):
        super(SoftXGBoost, self).__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = nn.ModuleList([SoftDecisionTree(input_dim, depth=tree_depth) for _ in range(n_estimators)])

    def forward(self, x):
        output = torch.zeros(x.size(0), 1).to(device)
        for tree in self.trees:
            output = output + self.learning_rate * tree(x)
        return output

# 训练与评估函数
def train_model(model, optimizer, criterion, x, y, n_epochs=300):
    model.train()
    loss_list = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
    return loss_list

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x)
        mse_loss = nn.MSELoss()(preds, y).item()
        rmse_loss = np.sqrt(mse_loss)
    return preds, mse_loss, rmse_loss

# 反标准化函数
def inverse_transform(y_scaled, scaler):
    if isinstance(y_scaled, np.ndarray):
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32).to(device)
    y_scaled = y_scaled.reshape(-1, 1)  # 确保是 2D 数组
    return scaler.inverse_transform(y_scaled.cpu().numpy())

# 训练10次模型，并在X_train和X_test中生成out_mod_'+str(i)列
num_models = 10
input_dim = X_train.shape[1]

# 初始化保存预测结果的 DataFrame
X_train_predictions = pd.DataFrame(X_train, columns=predictors)
X_test_predictions = pd.DataFrame(X_test, columns=predictors)

# 初始化保存预测结果的 DataFrame
# 不再需要 X_train_predictions 和 X_test_predictions

for i in range(num_models):
    print(f"Training model {i + 1}/{num_models}")
    
    # 初始化模型
    rf_model = SoftRandomForest(n_trees=10, tree_depth=3, input_dim=input_dim).to(device)
    xgb_model = SoftXGBoost(n_estimators=20, tree_depth=3, learning_rate=0.1, input_dim=input_dim).to(device)
    
    # 定义优化器和损失函数
    optimizer_rf = optim.Adam(rf_model.parameters(), lr=0.01)
    optimizer_xgb = optim.Adam(xgb_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练模型
    print(f"Training Soft Random Forest model {i + 1}...")
    train_model(rf_model, optimizer_rf, criterion, X_train_tensor, y_train_tensor, n_epochs=300)
    print(f"Training Soft XGBoost model {i + 1}...")
    train_model(xgb_model, optimizer_xgb, criterion, X_train_tensor, y_train_tensor, n_epochs=300)
    
    # 在训练集和测试集上进行预测
    rf_train_preds = rf_model(X_train_tensor).detach().cpu().numpy()
    rf_test_preds = rf_model(X_test_tensor).detach().cpu().numpy()
    xgb_train_preds = xgb_model(X_train_tensor).detach().cpu().numpy()
    xgb_test_preds = xgb_model(X_test_tensor).detach().cpu().numpy()
    
    # 反标准化预测值
    rf_train_preds = inverse_transform(rf_train_preds, scaler_y)
    rf_test_preds = inverse_transform(rf_test_preds, scaler_y)
    xgb_train_preds = inverse_transform(xgb_train_preds, scaler_y)
    xgb_test_preds = inverse_transform(xgb_test_preds, scaler_y)
    
    # 将预测值添加到 df_train 和 df_test 中
    df_train[f'rf_out_mod_{i}'] = rf_train_preds
    df_test[f'rf_out_mod_{i}'] = rf_test_preds
    df_train[f'xgb_out_mod_{i}'] = xgb_train_preds
    df_test[f'xgb_out_mod_{i}'] = xgb_test_preds

# 保存结果到文件（可选）
df_train.to_csv("F:/gemoxing/suijisenlin/df_train_predictions.csv", index=False)
df_test.to_csv("F:/gemoxing/suijisenlin/df_test_predictions.csv", index=False)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
df_train = pd.read_csv("F:/gemoxing/suijisenlin/df_train_predictions.csv")
df_test = pd.read_csv("F:/gemoxing/suijisenlin/df_test_predictions.csv")
# 假设 df_test 和 df_train 是已经处理好的测试集和训练集
# 假设 'tem' 是实际值列，'out_mod_i' 是预测值列

y_test = df_test['tem']
y_train = df_train['tem']

# 求多个模型的平均预测值
y_test_pred = df_test.filter(regex='xgb_out_mod_').mean(axis=1)
y_train_pred = df_train.filter(regex='xgb_out_mod_').mean(axis=1)

# 计算统计量
def calculate_stats(y_true, y_pred):
    md = (y_true - y_pred).mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return md, rmse, n, r2

# 计算测试集和训练集的统计量
md_test, rmse_test, n_test, r2_test = calculate_stats(y_test, y_test_pred)
md_train, rmse_train, n_train, r2_train = calculate_stats(y_train, y_train_pred)
import matplotlib.pyplot as plt
import matplotlib
plt.figure(figsize=(4, 8))  # 调整整体图的大小，确保有足够的空间放置两个正方形子图
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 未参与训练的数据集散点图
plt.subplot(2, 1, 1)  # 改为上下放置
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Unseen Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('d) Unseen Data (SoftXGBoost)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_test.min(), y_test.max(), f'MD: {md_test:.2f}\nRMSE: {rmse_test:.2f}\nN: {n_test}\nR^2: {r2_test:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

# 训练集散点图
plt.subplot(2, 1, 2)  # 改为上下放置
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train Date', color='orange')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('i) Train Data (SoftXGBoost)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_train.min(), y_train.max(), f'MD: {md_train:.2f}\nRMSE: {rmse_train:.2f}\nN: {n_train}\nR^2: {r2_train:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

plt.tight_layout()
plt.savefig('F:\gemoxing\moxingduibi\SoftXGBoost_scatter.png', dpi=600, bbox_inches='tight')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
df_train = pd.read_csv("F:/gemoxing/suijisenlin/df_train_predictions.csv")
df_test = pd.read_csv("F:/gemoxing/suijisenlin/df_test_predictions.csv")
# 假设 df_test 和 df_train 是已经处理好的测试集和训练集
# 假设 'tem' 是实际值列，'out_mod_i' 是预测值列

y_test = df_test['tem']
y_train = df_train['tem']

# 求多个模型的平均预测值
y_test_pred = df_test.filter(regex='rf_out_mod_').mean(axis=1)
y_train_pred = df_train.filter(regex='rf_out_mod_').mean(axis=1)

# 计算统计量
def calculate_stats(y_true, y_pred):
    md = (y_true - y_pred).mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return md, rmse, n, r2

# 计算测试集和训练集的统计量
md_test, rmse_test, n_test, r2_test = calculate_stats(y_test, y_test_pred)
md_train, rmse_train, n_train, r2_train = calculate_stats(y_train, y_train_pred)
import matplotlib.pyplot as plt
import matplotlib
plt.figure(figsize=(4, 8))  # 调整整体图的大小，确保有足够的空间放置两个正方形子图
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 未参与训练的数据集散点图
plt.subplot(2, 1, 1)  # 改为上下放置
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Unseen Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('c) Unseen Data (SoftRandomForest)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_test.min(), y_test.max(), f'MD: {md_test:.2f}\nRMSE: {rmse_test:.2f}\nN: {n_test}\nR^2: {r2_test:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

# 训练集散点图
plt.subplot(2, 1, 2)  # 改为上下放置
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train Date', color='orange')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('h) Train Data (SoftRandomForest)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_train.min(), y_train.max(), f'MD: {md_train:.2f}\nRMSE: {rmse_train:.2f}\nN: {n_train}\nR^2: {r2_train:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

plt.tight_layout()
plt.savefig('F:\gemoxing\moxingduibi\SoftRandomForest_scatter.png', dpi=600, bbox_inches='tight')
plt.show()
