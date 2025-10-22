import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子为一个固定的值
seed = 42
set_seed(seed)

# 读取数据
df = pd.read_csv("F:/zhandianshujvzhengli/df_y1n3.csv")
df.drop(['time_era'], axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'])  # 确保 date 列是 datetime 类型

# 1. 统计不同时间点的数量
unique_dates = df['date'].nunique()  # 获取不同时间点的数量
print(f"不同时间点的数量: {unique_dates}")

# 2. 按日期排序
df = df.sort_values(by='date')

# 3. 计算训练集、验证集和测试集的划分点
train_ratio = 0.86
val_ratio = 0.04
test_ratio = 0.10

# 计算划分点的索引
train_split_index = int(len(df) * train_ratio)
val_split_index = train_split_index + int(len(df) * val_ratio)

# 找到划分点的时间点
train_split_date = df.iloc[train_split_index]['date']
val_split_date = df.iloc[val_split_index]['date']

print(f"训练集划分点的时间点: {train_split_date}")
print(f"验证集划分点的时间点: {val_split_date}")

# 4. 划分数据集
df_train = df[df['date'] < train_split_date]
df_val = df[(df['date'] >= train_split_date) & (df['date'] < val_split_date)]
df_test = df[df['date'] >= val_split_date]

Predictors = (
    ["t2m_amp", "t2m_10y_avg", "ws_10y_avg", "sp_10y_avg", "t2m_mon_1", "ws_mon_1", "sp_mon_1", "t2m_mon_0", "ws_mon_0", "sp_mon_0"]
    + ["ws_" + str(i) for i in range(3)]
    + ["sp_" + str(i) for i in range(3)]
    + ["t2m_" + str(i) for i in range(3)]
    + ["dem", "month", "latitude"]
)  # 替换为实际的特征列名

TargetVariable = "tem"
if not all(col in df.columns for col in Predictors + [TargetVariable]):
    raise ValueError("Some columns are missing in the DataFrame.")

# 数据标准化
scaler = MinMaxScaler()
df_train_scaled = scaler.fit_transform(df_train[Predictors].values.astype('float64'))
df_val_scaled = scaler.transform(df_val[Predictors].values.astype('float64'))
df_test_scaled = scaler.transform(df_test[Predictors].values.astype('float64'))

# 创建数据集
X_train = df_train_scaled
y_train = df_train[TargetVariable].values
X_val = df_val_scaled
y_val = df_val[TargetVariable].values
X_test = df_test_scaled
y_test = df_test[TargetVariable].values

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 定义早停机制
class CustomEarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {self.best_score}')
        
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bidirectional = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(128 * 2, 64)  
        self.dense2 = nn.Linear(64, 8)
        self.dense3 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度，形状变为 (batch_size, 1, seq_len, features)
        x = F.relu(self.conv1(x))  # 形状变为 (batch_size, 64, seq_len, features)
        x = F.relu(self.conv2(x))  # 形状变为 (batch_size, 128, seq_len, features)
        x = self.maxpool(x.permute(0, 2, 1))  # 形状变为 (batch_size, seq_len/2, 128)
        x, _ = self.lstm1(x)  # 形状变为 (batch_size, seq_len/2, 128)
        x = self.dropout1(x)
        x, _ = self.bidirectional(x)  # 形状变为 (batch_size, seq_len/2, 128*2)
        x = x[:, -1, :]  # 取最后一个时间步的输出，形状变为 (batch_size, 128*2)
        x = self.dropout2(x)
        x = torch.sigmoid(self.dense1(x))
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# 计算 RMSE 的函数
def calculate_rmse(model, X, y_true, criterion):
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        y_pred = model(X).detach()  # 模型的预测值
        rmse = torch.sqrt(criterion(y_pred, y_true.unsqueeze(1).detach()))  # 确保维度一致
    return rmse.item(), y_pred.numpy()  # 返回 RMSE 和预测值

# 训练模型
def train_model_no_attention(X_train, y_train, X_val, y_val):
    # 定义早停机制
    early_stopping = CustomEarlyStopping(patience=10, verbose=True)

    # 定义模型
    Torchmodel = Net()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(Torchmodel.parameters(), lr=1e-3, weight_decay=1e-5)

    # 定义学习率调整器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 创建数据加载器
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

    # 存储训练和验证损失
    train_losses = []
    val_losses = []

    epochs = 300
    for epoch in range(epochs):
        Torchmodel.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            y_pred = Torchmodel(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))  # 确保维度一致
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loader)
        train_losses.append(np.sqrt(train_loss))

        Torchmodel.eval()
        val_rmse, _ = calculate_rmse(Torchmodel, X_val, y_val, criterion)
        val_losses.append(val_rmse)

        scheduler.step(val_rmse)
        early_stopping(val_rmse)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch % 10 == 0:
            print('*' * 10, 'Epoch: ', epoch, '\ttrain RMSE: ', np.sqrt(train_loss), '\tval RMSE', val_rmse)

    return Torchmodel, train_losses, val_losses

# 训练10次模型
num_models = 10
for i in range(num_models):
    print(f"Training model {i + 1}/{num_models}")
    # 训练模型
    model, train_losses, val_losses = train_model_no_attention(X_train, y_train, X_val, y_val)
    
    # 生成预测值
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).numpy()
        val_pred = model(X_val).numpy()
        test_pred = model(X_test).numpy()
    
    # 将预测值添加到对应的DataFrame中
    df_train[f'out_mod_{i}'] = train_pred
    df_val[f'out_mod_{i}'] = val_pred
    df_test[f'out_mod_{i}'] = test_pred
# 保存带有预测值的DataFrame到CSV文件
df_train.to_csv("F:/gemoxing/LSTM/df_train_with_predictions.csv", index=False)
df_val.to_csv("F:/gemoxing/LSTM/df_val_with_predictions.csv", index=False)
df_test.to_csv("F:/gemoxing/LSTM/df_test_with_predictions.csv", index=False)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df_train = pd.read_csv("F:/gemoxing/LSTM/df_train_with_predictions.csv")
df_val = pd.read_csv("F:/gemoxing/LSTM/df_val_with_predictions.csv")
df_test = pd.read_csv("F:/gemoxing/LSTM/df_test_with_predictions.csv")

# 将测试集和验证集合并为未参与训练的数据集
df_unseen = pd.concat([df_test, df_val])

# 提取实际值和预测值
y_train = df_train['tem']
y_unseen = df_unseen['tem']

# 求多个模型的平均预测值
y_train_pred = df_train.filter(regex='out_mod_').mean(axis=1)
y_unseen_pred = df_unseen.filter(regex='out_mod_').mean(axis=1)

# 计算统计量
def calculate_stats(y_true, y_pred):
    md = (y_true - y_pred).mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return md, rmse, n, r2

# 计算训练集和未参与训练的数据集的统计量
md_train, rmse_train, n_train, r2_train = calculate_stats(y_train, y_train_pred)
md_unseen, rmse_unseen, n_unseen, r2_unseen = calculate_stats(y_unseen, y_unseen_pred)
import matplotlib.pyplot as plt
import matplotlib
plt.figure(figsize=(4, 8))  # 调整整体图的大小，确保有足够的空间放置两个正方形子图
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 未参与训练的数据集散点图
plt.subplot(2, 1, 1)  # 改为上下放置
plt.scatter(y_unseen, y_unseen_pred, alpha=0.5, label='Unseen Data')
plt.plot([y_unseen.min(), y_unseen.max()], [y_unseen.min(), y_unseen.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('a) Unseen Data (LSTM-CNN)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_unseen.min(), y_unseen.max(), f'MD: {md_unseen:.2f}\nRMSE: {rmse_unseen:.2f}\nN: {n_unseen}\nR^2: {r2_unseen:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

# 训练集散点图
plt.subplot(2, 1, 2)  # 改为上下放置
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train Date', color='orange')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('f) Train Data (LSTM-CNN)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_train.min(), y_train.max(), f'MD: {md_train:.2f}\nRMSE: {rmse_train:.2f}\nN: {n_train}\nR^2: {r2_train:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

plt.tight_layout()
plt.savefig('F:\gemoxing\moxingduibi\LSTM_CNN_scatter.png', dpi=600, bbox_inches='tight')
plt.show()