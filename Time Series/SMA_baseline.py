# %%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf

from pmdarima import auto_arima
import statsmodels.api as sm

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# %%
def read_file(customer_file, site_id):
    """This function reads raw kWh report, generates dataframe for a site with corresponding customer name"""
    
    df = pd.read_csv(customer_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    
    customer_name = str(customer_file[0:-10])
    dff = df.loc[df['Site'] == site_id]
    
    return customer_name, site_id, dff

#Inputs
customer_name, site_id, data_daily = read_file('Caliber_Daily.csv', 'CCC0501')

# %%
def split_data(dataseries, column_name):
    """This function splits time-series data into train & test datasets and plots them"""
    
    min_value, max_value = dataseries['Date'].min(), dataseries['Date'].max()
    
    #an optimal timeframe is under exploration
    train = dataseries[column_name].loc[min_value:'2020-09-15']
    test = dataseries[column_name].loc['2020-09-16':'2020-11-19']
    
    plt.figure(figsize = (9, 6))
    plt.plot(train, color = 'b')
    plt.title([customer_name, site_id])
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (kWh)')
    plt.plot(test, color = 'orange')
    plt.legend(['Train', 'Test'])
    
    return train, test, plt.show()

# %%
#User input: Time-series data & univariate series name

train, test, plot = split_data(data_daily, 'Energy Consumption (kWh)')
round(((len(test)/len(train))*100), 2)

# %% [markdown]
# # Moving Average

# %%
#rolling window
window_size = 7
hist = [train[i] for i in range(len(train))]
pred0 = []
for t in range(len(test)):
    yhat = np.mean(hist[-(window_size)])
    obs = test.iloc[t]
    pred0.append(yhat)
    hist.append(obs)
    
#len(pred0)

# %%
#resultsdf = pd.DataFrame() #resultsdf['Forecast'] = 0

def check_error(orig, fore):
    """This function generates performance metrics"""
    
    mse = mean_squared_error(orig, fore)
    rmse = round(sqrt(mse), 2)
    mape = round(np.mean(np.abs((orig - fore) / orig)) * 100, 2)
    metrics = [rmse, mape]
    
    print('RMSE = ', metrics[0])
    print('MAPE = ', metrics[1])

    
#Inputs
check_error(test, pred0)

# %%
plt.plot(train.rolling(window = window_size).mean(), label = 'Forecast-SMA') 
plt.plot(train, label = 'Observed')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()




