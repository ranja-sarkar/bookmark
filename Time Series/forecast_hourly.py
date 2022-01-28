# %%
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

import time
import warnings
warnings.filterwarnings("ignore")

# %%
def read_file(customer_file, site_id):
    """This function reads raw kWh report, generates dataframe for a site with corresponding customer name"""
    
    df = pd.read_csv(customer_file)
    df['Date'] = pd.to_datetime(df['Start Time'])
    df = df.set_index(df['Date'])
    
    customer_name = str(customer_file[0:-15])
    dff = df.loc[df['Site ID'] == site_id]
    
    return customer_name, site_id, dff

# %%
#User inputs: Customer filename & Site ID

customer_name, site_id, data = read_file('Nouria_kWh_Report.csv', 'Nouria04029')

# %%
#Data-series autocorrelation

pd.plotting.autocorrelation_plot(data['Sum'])
plt.title([customer_name, site_id])
plt.show()

# %%
#Check data time-range; split it into train & test datasets (depending on the given range for a customer site)
#Forecasts here are created for 72hours/3days

min_value, max_value = data['Date'].min(), data['Date'].max()

train = data['Sum'].loc[min_value:'2020-11-17 00:00:00']
test = data['Sum'].loc['2020-11-17 01:00:00':'2020-11-20 00:00:00']

#Plot Split Data
plt.figure(figsize = (9, 6))
plt.plot(train, color = 'b')
plt.title([customer_name, site_id])
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.plot(test, color = 'orange')
plt.legend(['Train', 'Test'])
plt.show()


# %%
def adfuller_test(dataseries):
    """This function checks data stationarity"""
    
    adf = adfuller(dataseries)
    output = pd.Series(adf[0:4], index = ['ADF Statistic', 'p-value', 'Lags', '#comments used'])
    for key, value in adf[4].items():
        output["Critical Value (%s)" %key] = value
    return output


adfuller_test(train)

# %%
#TRAIN Model: AR-order selection depends on lags for stationarity & AIC

model = ARMA(train, order = [24, 0]).fit()
print(model.summary().tables[0])

#print(time.time()) #seconds

# %%
#TEST Model

preds = model.forecast(len(test))[0]

# %%
#Plot & compare predictions with test dataset

forecast_data = pd.DataFrame(preds, index = test.index, columns = ['Forecast'])
pd.concat([test, forecast_data], axis = 1).plot(title = site_id)
output = pd.concat([test, forecast_data], axis = 1)

# %%
def check_error(orig, fore):
    """This function generates performance metric"""
    
    resid = round(np.mean(orig - fore), 2)
    mse = mean_squared_error(orig, fore)
    rmse = round(sqrt(mse), 2)
    metrics = [resid, rmse]
    
    return print('RMSE =', metrics[1])

check_error(test, preds)


