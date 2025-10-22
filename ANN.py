# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:31:17 2025

@author: lenovo
"""
import rasterio
import pyproj  # 确保导入 pyproj 库
from pyproj import Transformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import rasterio as rio
import dask
import matplotlib
import rasterio
from rasterio.transform import xy
# loading T10m dataset
df = pd.read_csv("H:\wangshujing\df_y1n3_1.0.csv")

ice = gpd.GeoDataFrame.from_file("H:\wangshujing\greenland_land_3413/greenland_land_3413.shp")
ice = ice.to_crs("EPSG:3413")
ice_4326 = ice.to_crs(4326)
# %%
firn_memory = 3
Predictors = (
    ["t2m_amp", "t2m_10y_avg",
     "ws_10y_avg",
     "sp_10y_avg","t2m_mon_1","ws_mon_1","sp_mon_1","t2m_mon_0","ws_mon_0","sp_mon_0"] 
    + ["ws_" + str(i) for i in range(firn_memory)]
    + ["sp_" + str(i) for i in range(firn_memory)]
    + ["t2m_" + str(i) for i in range(firn_memory)]
    + ["dem"]+ ["month"]+ ["LAT"]
)


# %%# Extracting ERA5 data
print('loading ERA5 data')
produce_input_grid = 1
if produce_input_grid:
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
       ''' 
打开数据集
       '''
# 合并两个数据集
#era5_land
    ds_era = xr.open_dataset('H:\wangshujing\gelinglan\ERA5\\t2m\\t2m_1940_2024_d_sea_K.nc',engine='netcdf4')[['t2m']]
    ds_era['sp'] = xr.open_dataset('H:\wangshujing\gelinglan\ERA5\sp\sp_1940_2024_d_sea.nc',engine='netcdf4')['sp']
    ds_era['ws'] = xr.open_dataset('H:\wangshujing\gelinglan\ERA5\wind\wind_1940_2024_d_sea.nc',engine='netcdf4')['si10']
    ds_era['dem'] = xr.open_dataset('H:\wangshujing\GrIS_dem\dem_linear.nc',engine='netcdf4')['dem']
    ds_era['LAT'] = xr.open_dataset('H:\wangshujing\gelinglan\ERA5\sp\sp_1940_2024_d_sea.nc',engine='netcdf4')['latitude']
    
    ds_era['t2m'] = ds_era['t2m'] - 273.15
    ds_era['sp'] = ds_era['sp'] / 100
    
    print('t2m_avg and sp_avg')
    ds_era["t2m_10y_avg"] = ds_era.t2m.rolling(valid_time=10*12).mean()
    ds_era["sp_10y_avg"] = ds_era.sp.rolling(valid_time=10*12).mean()
    ds_era["ws_10y_avg"] = ds_era.ws.rolling(valid_time=10*12).mean()
    
    print('t2m_amp')
    ds_era["t2m_amp"] = ds_era.t2m.rolling(valid_time=12).max() - ds_era.t2m.rolling(valid_time=12).min()
    
    print('t2m_0 and sp_0')    
    ds_era["t2m_0"] = ds_era.t2m.rolling(valid_time=12).mean()
    ds_era["sp_0"] = ds_era.sp.rolling(valid_time=12).mean()
    ds_era["ws_0"] = ds_era.ws.rolling(valid_time=12).mean()
    #ds_era["ssr_0"] = ds_era.ssr.rolling(valid_time=12).mean()
    print('t2m_mon_0 and sp_mon_0')    
    ds_era["t2m_mon_0"] = ds_era["t2m"]
    ds_era["sp_mon_0"] = ds_era["sp"]
    ds_era["ws_mon_0"] = ds_era["ws"]
    #ds_era["ssr_0"] = ds_era.ssr.rolling(valid_time=12).mean()
    
    ds_era["t2m_mon_1"] = ds_era["t2m_mon_0"].shift(valid_time=12)
    ds_era["sp_mon_1"] = ds_era["sp_mon_0"].shift(valid_time=12)
    ds_era["ws_mon_1"] = ds_era["ws_mon_0"].shift(valid_time=12)
    
    for k in range(0, firn_memory):
        print(k+1,'/',firn_memory)
        ds_era["t2m_" + str(k)] = ds_era["t2m_0"].shift(valid_time=12*k)
        ds_era["sp_" + str(k)] = ds_era["sp_0"].shift(valid_time=12*k)
        ds_era["ws_" + str(k)] = ds_era["ws_0"].shift(valid_time=12*k)
        
    ds_era['month'] = np.cos((ds_era.valid_time.dt.month-1)/12*2*np.pi)#月份余弦值
    ds_era = ds_era.isel(valid_time=slice(10*12,None))


# %%

print('plotting')
fig, ax = plt.subplots(4, 6, figsize=(21, 15))
plt.subplots_adjust(hspace=0.6, wspace=0.3)  # 调整子图间距
ax = ax.flatten()  # 将二维数组展平为一维数组，方便后续操作

# 绘制23个子图
for i in range(23):
    ax[i].plot(df[Predictors[i]], df['tem'], 
               marker="o", linestyle="None", markersize=1.5)
    
    ax[i].set_xlabel(Predictors[i])
    ax[i].set_ylabel('tem', labelpad=-1)

# 隐藏最后一个多余的子图
ax[23].axis("off")

# 显示图形
plt.show()
# masking on ice sheet
ds_era = ds_era.rio.write_crs(4326)
ice_4326 = ice.to_crs(4326)
msk = ds_era.t2m_amp.isel(valid_time=12*5).rio.clip(ice_4326.geometry, ice_4326.crs)
ds_era = ds_era.where(msk.notnull())

# %% Representativity of the dataset
print('calculating weights based on representativity')
bins_temp = np.linspace(-40, 10.01, 15) 
bins_sp = np.linspace(650, 1050.01, 15)
bins_ws = np.linspace(0, 10.01, 15)
bins_amp = np.linspace(0, 55.01, 15)
# first calculating the target histograms (slow)
pred_list = ["t2m_10y_avg", "sp_10y_avg", "ws_10y_avg", "t2m_amp"]
target_hist = [None] * len(pred_list)
for i in range(len(pred_list)):
    c=1
    d=0
    if pred_list[i] == 't2m_10y_avg': 
        bins = bins_temp        
    if "ws" in pred_list[i]: bins = bins_ws
    if "amp" in pred_list[i]: bins = bins_amp
    if "sp" in pred_list[i]:         
        bins = bins_sp
    print(pred_list[i])
    target_hist[i], _ = np.histogram(ds_era[pred_list[i]].values*c + d, bins=bins)   
    target_hist[i] = target_hist[i].astype(np.float32) / target_hist[i].sum()

    hist1 = target_hist[i]
    hist2, _ = np.histogram(df[pred_list[i]].values*c + d, bins=bins)

    weights_bins = 0 * hist1 + 1
    weights_bins[hist2 != 0] = hist1[hist2 != 0] / hist2[hist2 != 0]
    ind = np.digitize(df[pred_list[i]].values*c + d, bins)
    df[pred_list[i] + "_w"] = weights_bins[ind - 1]
df["weights"] = df[[p + "_w" for p in pred_list]].mean(axis=1)

#  plotting histograms
abc='abcdefg'
pred_name = [r"$\overline{T_{2m,\ 10\ y}}$ (°C)",
             r"$\overline{SP_{10\ y}}$ (hPa)",
             r"$\overline{WS_{10\ y}}$ (m/s)",
             r"$T_{2m, amp.}$ (°C)"]
#计算加权中位数
def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median
h=['','','','']

fig, ax = plt.subplots(2, 4, figsize=(11, 8), sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.8, wspace=0.05, hspace=1)
for k in range(2):
    if k == 0:
        ttl = "All observations weigh equally"
    else:
        ttl = "With weights based on representativity"
    for i in range(len(pred_list)):
        c=1
        d=0
        if pred_list[i] == 't2m_10y_avg': 
            bins = bins_temp
        if "ws" in pred_list[i]: bins = bins_ws
        if "amp" in pred_list[i]: bins = bins_amp
        if "sp" in pred_list[i]: 
           bins = bins_sp

        hist1 = target_hist[i]

        if k == 0:
            hist2, _ = np.histogram(df[pred_list[i]].values*c + d, bins=bins)
        elif k == 1:
            hist2, _ = np.histogram(
                df[pred_list[i]].values*c + d, bins=bins, weights=df["weights"].values
            )

        hist2 = hist2.astype(np.float32) / hist2.sum()

        h[0] = ax[k,i].bar(bins[:-1], hist2,
            width=(bins[1] - bins[0]),
            alpha=0.5, color="tab:blue", edgecolor="lightgray",
            label="ERA5 values at the observation sites",
        )
        if k == 0:
            med1=np.median(df[pred_list[i]].values*c + d)
        else:
            med1=weighted_median(df[pred_list[i]].values*c + d, df["weights"].values)
        h[1] = ax[k,i].axvline(med1,
            color="tab:blue", ls='--', lw=3, label="median")
        h[2] = ax[k,i].bar(bins[:-1], hist1,
            width=(bins[1] - bins[0]),
            alpha=0.5, color="tab:orange", edgecolor="lightgray",
            label="ERA5 values for the entire land",
        )
        med2 =np.nanmedian(ds_era[pred_list[i]].values*c + d)
        print(pred_list[i]+' %0.1f'%(med1- med2))
        h[3] = ax[k,i].axvline(med2,
            color="tab:orange", ls='--', lw=3,
            label="median",
        )

        ind = (hist1 + hist2)>0
        d_canberra = np.sum(np.abs(hist1[ind] - hist2[ind]) / (hist1[ind]))
        ax[k,i].annotate(r"$d_{Canberra}= %0.1f$"% d_canberra,
            xy=(0.65, 0.89), xycoords="axes fraction")

        ax[k,i].set_ylim(0,0.35)
        ax[k,i].set_xlabel(pred_name[i],fontsize=14)
        if i ==0:
            ax[k,i].set_ylabel("Probability (-)",fontsize=14)
        else:
            ax[k,i].set_ylabel("")
        ax[k,i].set_title('(%s)'%abc[k*3+i], loc='left',weight='bold')
        ax[k,i].grid(axis='y')
        if i ==1:
            ax[k,i].legend(handles=h,
                           loc="lower center",
                           title=ttl,
                           title_fontproperties={'weight':'bold'},
                           ncol=2,
                           bbox_to_anchor=(0.5, 1.2))

# %% ANN functions
# definition of the useful function to fit an ANN and run the trained model

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.layers import GaussianNoise
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

def train_ANN(df, Predictors, TargetVariable="tem",
              num_nodes = 250, num_layers=3,
              epochs=250, batch_size=3000,plot=False):

    w = df["weights"].values
    X = df[Predictors].values
    y = df[TargetVariable].values.reshape(-1, 1)

    # Sandardization of data
    PredictorScalerFit = StandardScaler().fit(X)
    TargetVarScalerFit = StandardScaler().fit(y)

    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)

    # create and fit ANN model

    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(len(Predictors),)))
    for i in range(num_layers):
        model.add(Dense(units=num_nodes, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam",weighted_metrics=[])
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0, sample_weight=w)
    return model, PredictorScalerFit, TargetVarScalerFit

def ANN_predict(df, model, PredictorScalerFit, TargetVarScalerFit):
    X = PredictorScalerFit.transform(df.values)
    Y = model.predict(X, verbose=0)
    Predictions = pd.DataFrame(
        data=TargetVarScalerFit.inverse_transform(Y),
        index=df.index,
        columns=["temperaturePredicted"],
    ).temperaturePredicted
    #Predictions.loc[Predictions > 0] = 0
    return Predictions


def create_model(n_layers, num_nodes):
    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(len(Predictors),)))
    for i in range(n_layers):
        model.add(Dense(units=num_nodes, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# %% 选取训练集和测试集
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import rasterio as rio
import dask
import matplotlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

df_test = pd.DataFrame()
df_train = pd.DataFrame()

# 按站点名分组
grouped = df.groupby('name')

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

# 输出结果
print("原始数据集大小:", len(df))
print("验证集大小:", len(df_test))
print("训练集大小:", len(df_train)) 
# %% Testing batch size and epoch numbers
test_stability = 1
if test_stability:
    df_train = df_train.copy()  # 替换为训练集
    df_test = df_test.copy()  # 替换为验证集

    # 给定的层数和节点数
    layers = 3  # 替换为你的层数
    nodes = 250  # 替换为你的节点数

    num_models = 10
    model = [None] * num_models
    PredictorScalerFit = [None] * num_models
    TargetVarScalerFit = [None] * num_models

    # training the models
    for i in range(num_models):
        model[i], PredictorScalerFit[i], TargetVarScalerFit[i] = train_ANN(
            df_train, Predictors, num_nodes=nodes, num_layers=layers, epochs=250,
            batch_size=1000, plot=False
        )
        df_test['out_mod_' + str(i)] = ANN_predict(df_test[Predictors], 
            model[i], PredictorScalerFit[i], TargetVarScalerFit[i]).values
        df_train['out_mod_' + str(i)] = ANN_predict(df_train[Predictors], 
            model[i], PredictorScalerFit[i], TargetVarScalerFit[i]).values

df_test.to_csv("H:/wangshujing/output/df_test_predictions1.csv", index=False)
df_train.to_csv("H:/wangshujing/output/df_train_predictions1.csv", index=False)

#fig.savefig('figures/layers_and_nodes.png',dpi=300)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 假设 df_test 和 df_train 是已经处理好的测试集和训练集
# 假设 'tem' 是实际值列，'out_mod_i' 是预测值列

y_test = df_test['tem']
y_train = df_train['tem']

# 求多个模型的平均预测值
y_test_pred = df_test.filter(regex='out_mod_').mean(axis=1)
y_train_pred = df_train.filter(regex='out_mod_').mean(axis=1)

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

# 绘制散点图
plt.figure(figsize=(12, 6))

# 测试集散点图
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Test Set')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Set')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_test.min(), y_test.max(), f'MD: {md_test:.2f}\nRMSE: {rmse_test:.2f}\nN: {n_test}\nR^2: {r2_test:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')

# 训练集散点图
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train Set', color='orange')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Train Set')
plt.legend(loc='lower right')  # 将图例放置在右下角
plt.text(y_train.min(), y_train.max(), f'MD: {md_train:.2f}\nRMSE: {rmse_train:.2f}\nN: {n_train}\nR^2: {r2_train:.2f}',
         fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.tight_layout()
plt.show()

# %% best model
print('Training model on entire dataset')
best_model, best_PredictorScalerFit, best_TargetVarScalerFit = train_ANN(
    df, Predictors, num_nodes = 250, num_layers=3, epochs=250, batch_size=1000)
#predictions = ANN_predict(df[Predictors], best_model, best_PredictorScalerFit, best_TargetVarScalerFit)
# %%将ds_era中加入dem数据
#ds_era['dem'] = xr.open_dataset('H:\wangshujing\GrIS_dem\dem_linear.nc',engine='netcdf4')['dem']
# %% Predicting T10m over ERA5 dataset
predict = 0

if predict:    
    print("predicting T2m over entire ERA5 dataset")
    print('converting to dataframe')
    tmp = ds_era[Predictors].to_dataframe()
    tmp = tmp.loc[tmp.notnull().all(1),:][Predictors]
    
    print('applying ANN (takes ~45min on my laptop)')
    out = ANN_predict(tmp, best_model, best_PredictorScalerFit, best_TargetVarScalerFit)

    ds_T2m =  out.to_frame().to_xarray()["temperaturePredicted"].sortby(['valid_time','latitude','longitude'])
    ds_T2m = ds_T2m.rio.write_crs(4326).rio.clip(ice.to_crs(4326).geometry).rename('T2m')
    ds_T2m.attrs['author'] = 'Shujing Wang'
    ds_T2m.attrs['contact'] = '17763191701@163.com'
    ds_T2m.attrs['description'] = 'Monthly grids of Greenland 2m surface temperature for 1950-2024 as predicted by an artifical neural network trained on more than 20000 in situ observations and using ERA5 surface pressure, wind speed, and 2m temperature as input.'
    ds_T2m.to_netcdf("H:\wangshujing\output\\T2m_prediction.nc", mode='w')
    
ds_T2m = xr.open_dataset("H:\wangshujing\output\\T2m_prediction.nc")["T2m"]


# %% spatial cross-validation:
print("fitting cross-validation models")
zwally = gpd.GeoDataFrame.from_file("H:/wangshujing/area_divied/fenqu/fenqu.shp")
print("Initial CRS of zwally:", zwally.crs)
zwally = zwally.to_crs(epsg=4326)
df = df.reset_index(drop=True)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
).set_crs(4326)
points_within = gpd.sjoin(gdf, zwally, op="within")


df["zwally_zone"] = np.nan
df["T2m_pred_unseen"] = np.nan
model_list = [None] * zwally.shape[0]
PredictorScalerFit_list = [None] * zwally.shape[0]
TargetVarScalerFit_list = [None] * zwally.shape[0]

for i in range(zwally.shape[0]):
    msk = points_within.loc[points_within.index_right == i, :].index
    df.loc[msk, "zwally_zone"] = i
    print(i, len(msk), "%0.2f" % (len(msk) / df.shape[0] * 100))

    df_cv = df.loc[df.zwally_zone != i, ].copy()

    model_list[i], PredictorScalerFit_list[i], TargetVarScalerFit_list[i] \
                            = train_ANN(df_cv, Predictors, 
                                           TargetVariable="tem",  
                                           num_nodes = 250, 
                                           num_layers=2, 
                                           epochs=250)
    # saving the model estimate for the unseen data
    df_unseen = df.loc[df.zwally_zone == i, :].copy()

    df.loc[df.zwally_zone == i,"T2m_pred_unseen"] = ANN_predict(df_unseen[Predictors],
                                              model_list[i],
                                              PredictorScalerFit_list[i],
                                              TargetVarScalerFit_list[i])

# defining function that makes the ANN estimate from predictor raw values
df = df.sort_values("date")

def ANN_model_cv(df_pred, model_list, PredictorScalerFit_list, TargetVarScalerFit_list):
    pred = pd.DataFrame()
    for i in range(zwally.shape[0]):
        pred["T2m_pred_" + str(i)] = ANN_predict(df_pred,
                                                  model_list[i],
                                                  PredictorScalerFit_list[i],
                                                  TargetVarScalerFit_list[i])
    df_mean = pred.mean(axis=1)
    df_std = pred.std(axis=1)
    return df_mean.values, df_std.values

# %% Predicting T10m uncertainty
predict = 1
if predict:
    print("predicting T2m uncertainty over entire ERA5 dataset")
    print("...takes about 10 hours")
    for year in range(1950,2025): #np.unique(ds_era.time.dt.year):
        ds_T2m_std = ds_era["t2m"].sel(valid_time=str(year)).copy().rename("T2m_std") * np.nan
        for valid_time in ds_T2m_std.valid_time:
            if (valid_time.dt.month==1) & (valid_time.dt.day==1):
                print(valid_time.dt.year.values)
            tmp = ds_era.sel(valid_time=valid_time).to_dataframe()
            _, tmp["cv_std"] = ANN_model_cv(tmp[Predictors], model_list,
                                            PredictorScalerFit_list,
                                            TargetVarScalerFit_list)
    
            ds_T2m_std.loc[dict(valid_time=valid_time)] = (
                tmp["cv_std"].to_frame().to_xarray()["cv_std"]
                .transpose("latitude", "longitude").values
                )
    
        ds_T2m_std.to_netcdf("H:/wangshujing/output/quyujiaocha/predicted_T2m_std_"+str(year)+".nc")
    ds_T2m_std = xr.open_mfdataset(
        ["H:/wangshujing/output/quyujiaocha/predicted_T2m_std_"+str(year)+".nc" for year in range(1950,2025)])
    ds_T2m_std = ds_T2m_std.rio.write_crs(4326)
    ice_4326 = ice.to_crs(4326)
    msk = ds_T2m_std.T2m_std.isel(valid_time=0).rio.clip(ice_4326.geometry, ice_4326.crs)
    ds_T2m_std = ds_T2m_std.where(msk.notnull())
    ds_T2m_std.attrs['author'] = 'Shujing Wang'
    ds_T2m_std.attrs['contact'] = 'bav@geus.dk'
    ds_T2m_std.attrs['description'] = 'Monthly grids of the uncertainty of the ANN Greenland 2m surface temperature for 1950-2024 calculated from the standard deviation between the predictions of 10 spatial cross-validation models.'
    ds_T2m_std.to_netcdf("H:/wangshujing/output/T2m_uncertainty_quyu.nc")
ds_T2m_std = xr.open_dataset("H:/wangshujing/output/T2m_uncertainty_quyu.nc")['T2m_std']


