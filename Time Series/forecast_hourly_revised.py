# %%
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv('Nouria_kWh_Report.csv')

df['Date'] = pd.to_datetime(df['Start Time'])
df = df.set_index(df['Date'])

dff = df.loc[df['SiteID'] == 'Nouria04029']
dff.shape

# %%
x = dff['Date']
y = dff['Energy Consumption (kWh)']
data = pd.DataFrame({'Date': x, 'Energy Consumption (kWh)':y})

#plt.figure(figsize = (12, 8))
##data['Energy Consumption (kWh)'].plot(title = "Energy Consumption (kWh)", label = '')
#plt.grid()
#plt.legend()

# %%
pd.plotting.autocorrelation_plot(y)
#Default is lag = 1
shift1 = y.autocorr()
plt.show()
print('Autocorrelation is', shift1.round(4))

# %%
#Specify lags

plot_acf(y)
plot_pacf(y)
plt.show()

# %%
round(y.mean(), 2)

# %%
dff['Date'].min(), dff['Date'].max()

# %%
train = dff['Energy Consumption (kWh)'].loc['2020-03-13 01:00:00':'2020-10-31 00:00:00']
test = dff['Energy Consumption (kWh)'].loc['2020-10-31 01:00:00':'2020-11-20 00:00:00']

# %%
len(train), len(test), round(train.mean(), 2)

# %%
training_site = pd.DataFrame('Nouria04029', index = train.index, columns = ['Site ID'])
training = pd.concat([train, training_site], axis = 1)
#len(training)
#training.to_csv('Nouria_training_data.csv')

# %%
plt.figure(figsize = (10, 8))
plt.plot(train, color = 'b')
plt.plot(test, color = 'orange')
plt.title('Nouria04029')
plt.legend(['Train', 'Test'])
plt.show()

# %%
train.hist()
plt.title('Count vs. kWh')

# %%
#Summary statistics

s = train.values
split = int(len(s)/2)
s1, s2 = s[0:split], s[split:]
mean1, mean2 = s1.mean(), s2.mean()
var1, var2 = s1.var(), s2.var()

print('Avg1 = %f, Avg2 = %f' % (mean1, mean2))
print('Var1 = %f, Var2 = %f' % (var1, var2))

# %%
result = seasonal_decompose(train, model = 'additive', period = 24)
fig = result.plot()
fig.set_size_inches(12, 8)

# %%
#Statistical Test

def adfuller_test(serie, figsize = (12,4), plot = True, title = ""):
#    if plot:
#        serie.plot(figsize = figsize, title = title)
#        plt.show()
    #Dickey Fuller test on the first difference
    adf = adfuller(serie)
    output = pd.Series(adf[0:4], index = ['ADF Statistic', 'p-value', 'Used Lags', 'Number of comments used'])
#    output = round(output, 4)
    
    for key, value in adf[4].items():
        output["Critical Value (%s)" %key] = value
    return output

#adfuller_test(train.diff().dropna(), title = '')
adfuller_test(train, title = '')

# %%
#ARMA(p,q) 

model = ARMA(train, order = [26, 0]).fit()
#pred = model.predict(start = len(train), end = (len(train)-1) + len(test))

# %%
#len(pred)
print(model.summary().tables[0])

# %%
preds, stderr, ci = model.forecast(len(test))

# %%
forecast_data = pd.DataFrame(preds, index = test.index, columns = ['Forecasted Energy Consumption'])
pd.concat([test, forecast_data], axis = 1).plot(title = 'Nouria04029')

# %%
compare = pd.concat([test, forecast_data], axis = 1)
compare.head(len(test))

# %%
#data = train.append([test])
#pd.concat([data, forecast_data], axis = 1).plot()

# %%
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

def check_error(orig, fore):
    
    resid = round(np.mean(orig - fore), 2)
    mse = mean_squared_error(orig, fore)
    rmse = round(sqrt(mse), 2)
        
    error_group = [resid, rmse]
#    serie = pd.DataFrame(error_group, columns = ['RESID','RMSE'])
#    serie.index.name = index_name
    
    return error_group

check_error(test, preds)

# %%
round(preds.mean(), 2), round(test.mean(), 2)


