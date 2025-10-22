import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load data
data = pd.read_csv("F:/zhandianshujvzhengli/df_y1n3.csv")

# Define parameters
firn_memory = 3
predictors = (
    ["t2m_amp", "t2m_10y_avg",
     "ws_10y_avg",
     "sp_10y_avg","t2m_mon_1","ws_mon_1","sp_mon_1","t2m_mon_0","ws_mon_0","sp_mon_0"]
     #"ssr_10y_avg"]
    + ["ws_" + str(i) for i in range(firn_memory)]
    + ["sp_" + str(i) for i in range(firn_memory)]
    + ["t2m_" + str(i) for i in range(firn_memory)]
    + ["dem"]+ ["month"]+["latitude"]
)

# Prepare data
X = data[predictors]
y = data["tem"]

# Split data into train (60%), validation (20%), and test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.14, random_state=seed_value)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7143, random_state=seed_value)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN (samples, timesteps, features)
# We'll treat the rr-year sequences as timesteps
n_features = len(predictors) // (3 * firn_memory + 1)  # 3 variables (smb, t2m, prep) + static features
time_steps = firn_memory

# Separate static and temporal features
static_features = ["t2m_amp", "t2m_10y_avg","ws_10y_avg","sp_10y_avg","month","dem","latitude",
                   't2m_mon_1', 'ws_mon_1', 'sp_mon_1', 't2m_mon_0', 'ws_mon_0', 'sp_mon_0']
temporal_features = [f for f in predictors if f not in static_features]

# Create separate arrays for static and temporal features
X_train_static = X_train[static_features].values
X_train_temporal = X_train[temporal_features].values.reshape(-1, 3, 3)  # 3 variables
X_train_temporal = X_train_temporal.transpose(0, 2, 1)

X_val_static = X_val[static_features].values
X_val_temporal = X_val[temporal_features].values.reshape(-1, 3, time_steps)
X_val_temporal = X_val_temporal.transpose(0, 2, 1)

X_test_static = X_test[static_features].values
X_test_temporal = X_test[temporal_features].values.reshape(-1, 3,time_steps)
X_test_temporal = X_test_temporal.transpose(0, 2, 1)

# Standardize static features
static_scaler = StandardScaler()
X_train_static_scaled = static_scaler.fit_transform(X_train_static)
X_val_static_scaled = static_scaler.transform(X_val_static)
X_test_static_scaled = static_scaler.transform(X_test_static)

# Standardize temporal features
temporal_scaler = StandardScaler()
X_train_temporal_scaled = temporal_scaler.fit_transform(X_train_temporal.reshape(-1, 3))
X_train_temporal_scaled = X_train_temporal_scaled.reshape(-1, time_steps, 3)

X_val_temporal_scaled = temporal_scaler.transform(X_val_temporal.reshape(-1, 3))
X_val_temporal_scaled = X_val_temporal_scaled.reshape(-1, time_steps, 3)

X_test_temporal_scaled = temporal_scaler.transform(X_test_temporal.reshape(-1, 3))
X_test_temporal_scaled = X_test_temporal_scaled.reshape(-1, time_steps, 3)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_model(static_shape, temporal_shape, head_size=32, num_heads=4, ff_dim=64, dropout=0.2):
    # Static features branch
    static_input = Input(shape=(static_shape[1],))
    static_dense = Dense(64, activation='relu')(static_input)
    static_dense = Dropout(dropout)(static_dense)
    
    # Temporal features branch (CNN + Transformer)
    temporal_input = Input(shape=temporal_shape[1:])
    
    # CNN part
    cnn = Conv1D(64, 3, activation='relu', padding='same')(temporal_input)
    cnn = Dropout(dropout)(cnn)
    cnn = Conv1D(64, 3, activation='relu', padding='same')(cnn)
    cnn = Dropout(dropout)(cnn)
    
    # Transformer part
    transformer = transformer_encoder(cnn, head_size, num_heads, ff_dim, dropout)
    transformer = GlobalMaxPooling1D()(transformer)
    
    # Combine branches
    combined = tf.keras.layers.concatenate([static_dense, transformer])
    dense = Dense(128, activation='relu')(combined)
    dense = Dropout(dropout)(dense)
    output = Dense(1)(dense)
    
    model = Model(inputs=[static_input, temporal_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

# Number of models to train
num_models = 10  # 可以根据需要调整模型数量

# Train multiple models
for i in range(num_models):
    print(f"Training model {i + 1}/{num_models}")
    
    # Build model
    model = build_model(
        static_shape=X_train_static_scaled.shape,
        temporal_shape=X_train_temporal_scaled.shape
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        [X_train_static_scaled, X_train_temporal_scaled],
        y_train_scaled,
        validation_data=([X_val_static_scaled, X_val_temporal_scaled], y_val_scaled),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Predict on test, validation, and train sets
    test_pred = model.predict([X_test_static_scaled, X_test_temporal_scaled]).flatten()
    val_pred = model.predict([X_val_static_scaled, X_val_temporal_scaled]).flatten()
    train_pred = model.predict([X_train_static_scaled, X_train_temporal_scaled]).flatten()
    
    # Store predictions in the respective DataFrames
    X_test[f'out_mod_{i}'] = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    X_val[f'out_mod_{i}'] = y_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
    X_train[f'out_mod_{i}'] = y_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()

# Save the DataFrames with predictions
X_test.to_csv("F:/gemoxing/ten+cnn/X_test_predictions.csv", index=False)
X_val.to_csv("F:/gemoxing/ten+cnn/X_val_predictions.csv", index=False)
X_train.to_csv("F:/gemoxing/ten+cnn/X_train_predictions.csv", index=False)
y_test.to_csv("F:/gemoxing/ten+cnn/y_test_predictions.csv", index=False)
y_val.to_csv("F:/gemoxing/ten+cnn/y_val_predictions.csv", index=False)
y_train.to_csv("F:/gemoxing/ten+cnn/y_train_predictions.csv", index=False)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
X_test = pd.read_csv("F:/gemoxing/ten+cnn/X_test_predictions.csv")
X_val = pd.read_csv("F:/gemoxing/ten+cnn/X_val_predictions.csv")
X_train = pd.read_csv("F:/gemoxing/ten+cnn/X_train_predictions.csv")
y_test = pd.read_csv("F:/gemoxing/ten+cnn/y_test_predictions.csv")
y_val = pd.read_csv("F:/gemoxing/ten+cnn/y_val_predictions.csv")
y_train = pd.read_csv("F:/gemoxing/ten+cnn/y_train_predictions.csv")

# 确保 y_not_train 和 y_train 是单列的 Series
y_test = y_test['tem']  # 假设 'tem' 是实际值列
y_val = y_val['tem']
y_train = y_train['tem']

# 检查并删除重复索引
X_test = X_test[~X_test.index.duplicated(keep='first')]
X_val = X_val[~X_val.index.duplicated(keep='first')]
y_test = y_test[~y_test.index.duplicated(keep='first')]
y_val = y_val[~y_val.index.duplicated(keep='first')]

# 检查并修改重复列名
X_test.columns = [f'{col}_{i}' for i, col in enumerate(X_test.columns)]
X_val.columns = [f'{col}_{i}' for i, col in enumerate(X_val.columns)]
X_train.columns = [f'{col}_{i}' for i, col in enumerate(X_train.columns)]

# 合并测试集和验证集
X_not_train = pd.concat([X_test, X_val]).reset_index(drop=True)
y_not_train = pd.concat([y_test, y_val]).reset_index(drop=True)

# 提取实际值
y_not_train = y_not_train.reset_index(drop=True)  # 重置索引
y_train = y_train.reset_index(drop=True)  # 重置索引

# 求多个模型的平均预测值
y_not_train_pred = X_not_train.filter(regex='out_mod_').mean(axis=1)
y_train_pred = X_train.filter(regex='out_mod_').mean(axis=1)

# 计算统计量
def calculate_stats(y_true, y_pred):
    md = (y_true - y_pred).mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return md, rmse, n, r2

# 计算未参与训练的数据集和训练集的统计量
md_not_train, rmse_not_train, n_not_train, r2_not_train = calculate_stats(y_not_train, y_not_train_pred)
md_train, rmse_train, n_train, r2_train = calculate_stats(y_train, y_train_pred)


import matplotlib.pyplot as plt
import matplotlib
# 绘制散点图
plt.figure(figsize=(4, 8))  # 调整整体图的大小，确保有足够的空间放置两个正方形子图
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 未参与训练的数据集散点图
plt.subplot(2, 1, 1)  # 改为上下放置
plt.scatter(y_not_train, y_not_train_pred, alpha=0.5, label='Unseen Data')
plt.plot([y_not_train.min(), y_not_train.max()], [y_not_train.min(), y_not_train.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('b) Unseen Data (Transformer-CNN)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_not_train.min(), y_not_train.max(), 
         f'MD: {md_not_train:.2f}\nRMSE: {rmse_not_train:.2f}\nN: {n_not_train}\nR^2: {r2_not_train:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

# 训练集散点图
plt.subplot(2, 1, 2)  # 改为上下放置
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train Date', color='orange')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Temperature (℃)')
plt.ylabel('Predicted Temperature (℃)')
plt.title('g) Train Data (Transformer-CNN)', loc='left')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_train.min(), y_train.max(), 
         f'MD: {md_train:.2f}\nRMSE: {rmse_train:.2f}\nN: {n_train}\nR^2: {r2_train:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.gca().set_aspect('equal', adjustable='box')  # 设置子图为正方形

plt.tight_layout()
plt.savefig('F:\gemoxing\moxingduibi\TensorFlow-CNN_scatter.png', dpi=600, bbox_inches='tight')
plt.show()