# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:14:49 2021

@author: admin
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

funFlag = 1
folderName='G:\Code_Python/'
dirList = os.listdir(folderName)
df = pd.read_csv('G:\Code_Python\Predict_test.csv')

df.head()
df.isnull().sum()

df = df.loc[df['Cua_hang'].notnull()]
df["Ngay"]=pd.to_datetime(df["Ngay"])

df=df.groupby('Ngay').sum()

df.plot()

seasonal_decompose(df,model='additive',freq=52).plot();

train=df[:1134] 
test=df[1134:] 

hwmodel=ExponentialSmoothing(train["So_luong"], seasonal='add', seasonal_periods=52).fit()
test_pred=hwmodel.forecast(7)

train['So_luong'].plot(legend=True, label='Train', figsize=(10,6))
test['So_luong'].plot(legend=True, label='Test')

test_pred.plot(legend=True, label='predicted_test')

np.sqrt(mean_squared_error(test,test_pred))

y_hat_avg = test.copy()
model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
y_hat_avg['Holt_Winter'] = model_fit.forecast(7)

#Function MAPE
def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.mean(np.abs((y - yhat) / y)) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape

rmse_opt=  np.sqrt(mean_squared_error(test["So_luong"], y_hat_avg['Holt_Winter']))
mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
mape_opt

def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.mean(np.abs((y - yhat) / y)) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape


'''Draft1'''
best_MAPE = np.inf
for j in range(len(df)):
    print(j)
    try:
        train=df[:j]
        test=df[j:j+7] 
        y_hat_avg = test.copy()
        model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
        y_hat_avg['Holt_Winter'] = model_fit.forecast(7)
        mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
        if mape_opt < best_MAPE:
            best_MAPE = mape_opt
    except:
       continue
print(best_MAPE)

j=52
train=df[:j]
test=df[j:j+7] 
y_hat_avg = test.copy()
model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
y_hat_avg['Holt_Winter'] = model_fit.forecast(7)
mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
print('Mean absolute percentage error:', round(mape_opt, 3))

'''draft2'''
best_MAPE = np.inf
for j in range(10):
    print(j)
    try:
        train=df[:j]
        test=df[j:j+7] # from aug19
        y_hat_avg = test.copy()
        model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
        y_hat_avg['Holt_Winter'] = model_fit.forecast(7)
        mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
        if mape_opt < best_MAPE: 
            best_MAPE = mape_opt
            y_hat_avg=y_hat_avg[j]
    except:
       continue
print(best_MAPE)

'''Chọn tập train tốt nhất từ ngày đầu tiên -> số lượng ngày nhất định & predict cho 7 ngày tiếp theo'''
i=1082
best_MAPE = np.inf
for j in range(606,i,1):
    print("j=",j)
    try:
        train=df[:j]
        test=df[j:j+7] # from aug19
        y_hat_avg = test.copy()
        model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
        y_hat_avg['Holt_Winter'] = model_fit.forecast(7)
        mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
        if mape_opt < best_MAPE:
            best_MAPE = mape_opt
            cfg=[j,best_MAPE]
            print('Mean absolute percentage error:', round(best_MAPE, 3))
    except:
       continue
print('Best Mean absolute percentage error:', round(best_MAPE, 3))

#-> j=606 là tốt nhất:5.11%

'''Chọn tập train tốt nhất từ 20/04/2021 trở về trước & predict cho 7 ngày tiếp theo'''
max=1082
y=1082
x=768
best_MAPE = np.inf
for j in range(x,y,1):
    print("j=",j)
    try:
        train=df[j:1134]
        test=df['2021-04-29':] # from aug19
        y_hat_avg = test.copy()
        model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
        y_hat_avg['Holt_Winter'] = model_fit.forecast(7)
        mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
        if mape_opt < best_MAPE:
            best_MAPE = mape_opt
            cfg=[j,best_MAPE]
            print('Mean absolute percentage error:', round(best_MAPE, 3))
    except:
       continue
print('Best Mean absolute percentage error:', round(best_MAPE, 3))

#-> j=769 là tốt nhất: 99.15%

'''Chọn tập train tốt nhất từ ngày đầu tiên -> số lượng ngày nhất định & predict đến ngày cuối cùng'''
y=800
x=700
best_MAPE = np.inf
for j in range(x,y,1):
    print("j=",j)
    try:
        train=df[:j]
        test=df[j:] # from aug19
        y_hat_avg = test.copy()
        model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
        y_hat_avg['Holt_Winter'] = model_fit.forecast(1141-j)
        mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
        if mape_opt < best_MAPE:
            best_MAPE = mape_opt
            cfg=[j,best_MAPE]
            print('Mean absolute percentage error:', round(best_MAPE, 3))
    except:
       continue
print('Best Mean absolute percentage error:', round(best_MAPE, 3))

# -> j=737 tốt nhất sau j=384: 64,6 (na ná nhau)


'''Chọn tập train tốt nhất từ ngày bất kỳ trong sample -> predict đến ngày cuối cùng'''
y=607
x=606
best_MAPE = np.inf
for j in range(x,y,1):
    for i in range(1,j-51,1):
        print("i=",i, "; ","j=",j)
        try:
            train=df[i:j]
            test=df[j:] # from aug19
            y_hat_avg = test.copy()
            model_fit = ExponentialSmoothing(np.asarray(train['So_luong']) ,seasonal_periods = 52 , seasonal='add').fit()
            y_hat_avg['Holt_Winter'] = model_fit.forecast(1141-j)
            mape_opt= MAPE(test["So_luong"],y_hat_avg['Holt_Winter'])
            if mape_opt < best_MAPE:
                best_MAPE = mape_opt
                cfg=[i,j,best_MAPE]
                print('Mean absolute percentage error:', round(best_MAPE, 3))
        except:
            continue
print('Best Mean absolute percentage error:', round(best_MAPE, 3))

m=best_1=np.inf
cfg = [m,best_1]
