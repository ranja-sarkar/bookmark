
import itertools
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas as pd
import statsmodels.api as sm
import matplotlib
import datetime

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
## matplotlib.rcParams['text.color'] = 'k'


# %%
df = pd.read_csv("C:/Users/ransarkar/Desktop/Trucks-Sales.csv")
#df.head(5)

#df['Trucks Sold'].min(),df['Trucks Sold'].max()
df['datetime'] = pd.to_datetime(df['Date(mm-dd-yy)'])
df = df.set_index('datetime')
df.head(5)


## df = df.sort_values('datetime')
cdf = df.drop('Date(mm-dd-yy)',axis=1)
cdf.head(5)

y1 = cdf['Trucks Sold'].resample('MS').mean()
y = y1.round(2)
## y.head(10)
y.replace(np.nan,0)
y.head(10)

y.plot(figsize=(12, 6))
plt.show()


from pylab import rcParams
rcParams['figure.figsize'] = 14, 7

decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
fig = decomposition.plot()
plt.show()

 
from pyramid.arima import auto_arima

stepwise_model = auto_arima(y, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True,
                           error_action='ignore', suppress_warnings=True, stepwise=True)

print(stepwise_model.aic(),stepwise_model.bic())


# %%
## Fit ARIMA: order=(3, 1, 0) seasonal_order=(1, 1, 0, 12)
mod = sm.tsa.statespace.SARIMAX(y,order=(3, 1, 0),seasonal_order=(1, 1, 0, 12))
results = mod.fit()
 
print(results.summary().tables[1])
## results.summary()

results.plot_diagnostics(figsize=(18, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
#pred_ci
 
ax = y['2014':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='Forecasted', alpha=.6, figsize=(16, 6))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Trucks sales quantity')
plt.legend()
plt.show()

##MODEL performance metric
y_forecasted = pred.predicted_mean
y_observed = y['2017-01-01':]
mse = ((y_forecasted - y_observed) ** 2).mean()

## print('The MSE of our forecasts is {}'.format(round(mse, 2)))
print('The RMSE of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=24)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='Observed', figsize=(16, 6))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Trucks sales quantity')
plt.legend()
plt.show()




