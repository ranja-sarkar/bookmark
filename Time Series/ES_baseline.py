# %%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

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
customer_name, site_id, data_daily = read_file('Caliber_Daily.csv', 'CCC0611')

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

# %%
#The average data period is 1/α. Ex. α = 1.0 -> lag = 1 period; α = 0.25 -> lag = 4 periods and so on.

alpha = [0.25, 0.5, 1.0]
for key, value in enumerate(alpha):
    model = ES(train).fit(smoothing_level = value)
    results = model.predict(start = 0, end = len(train))
    results.plot(figsize = (24, 8), title = 'ES with multiple lags', label = value)
train.plot(figsize = (24, 8), label = 'Train set')
plt.ylabel('kWh')
plt.legend()
plt.show()

# %%
#resultsdf = pd.DataFrame() #resultsdf['Forecast'] = 0

alpha = 0.25
model = ES(train).fit(smoothing_level = alpha)
resultsdf = model.predict(start = len(train), end = (len(train)-1) + len(test))

def check_error(orig, fore):
    """This function generates performance metrics"""
    
    mse = mean_squared_error(orig, fore)
    rmse = round(sqrt(mse), 2)
    mape = round(np.mean(np.abs((orig - fore) / orig)) * 100, 2)
    metrics = [rmse, mape]
    
    print('RMSE = ', metrics[0])
    print('MAPE = ', metrics[1])

    
#Inputs
check_error(test, resultsdf)


