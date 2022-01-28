# %%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
#import seaborn

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.tsa.arima_model import ARIMA

from pmdarima import auto_arima
import statsmodels.api as sm

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# %%
def sample_daily(customer_file):
    """This function creates daily data for all customer-sites in the hourly energy consumption data"""
    
    df = pd.read_csv(customer_file, parse_dates = ['Start Time'])
    df.set_index('Start Time', inplace = True)
    df['Date'] = df.index
    customer_name = str(customer_file[:-15])
    
    site_id = df['SiteID'].unique()
    site_name = df['Site Name'].unique()

    resultsdf = pd.DataFrame(columns = ['Site', 'Site Name', 'Energy Consumption (kWh)'])
    results = pd.DataFrame(columns = ['Site', 'Site Name', 'Energy Consumption (kWh)'])
    for i in range(0, len(site_id)):
        
        resultsdf['Energy Consumption (kWh)'] = df['Energy Consumption (kWh)'][df['SiteID'] == site_id[i]].resample('D').sum()
        resultsdf['Site'] = df['SiteID'][df['SiteID'] == site_id[i]].resample('D').first()
        resultsdf['Site Name'] = df['Site Name'][df['Site Name'] == site_name[i]].resample('D').first()
        results = results.append(resultsdf)
        results['Date'] = results.index      
 
    return results


# %%
#User input: Raw kWh report csv file (/data/raw/kWh Reports/)

data_daily = sample_daily('Caliber_kWh_Report.csv')
data_daily.to_csv('Caliber_Daily.csv')

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
def data_AC(column_name):
    """This function finds auto-correlation of the univariate series"""
    
    y = data_daily[column_name]
    pd.plotting.autocorrelation_plot(y)
    plt.title([customer_name, site_id])
    
    return plt.show()

# %%
#User input: Univariate series name

data_AC('Energy Consumption (kWh)')

# %%
def split_data(dataseries, column_name):
    """This function splits time-series data into train & test datasets and plots them"""
    
    min_value, max_value = dataseries['Date'].min(), dataseries['Date'].max()
    
    #an optimal timeframe is under exploration
    train = dataseries[column_name].loc[min_value:'2020-10-15']
    test = dataseries[column_name].loc['2020-10-16':'2020-11-19']
    
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

# %%
def data_decomposition(dataseries, freq):
    """This function plots components of the dataset"""
    
    result = seasonal_decompose(dataseries, period = freq)
    fig = result.plot().set_size_inches(12, 9)
    
    return fig

# %%
#User inputs: Data & repeat-pattern frequency

data_decomposition(train, 7)

# %%
def adfuller_test(dataseries):
    """This checks data stationarity (Dickey-Fuller Test) and prints the reults"""
    
    adf = adfuller(dataseries)
    output = pd.Series(adf[0:4], index = ['ADF Statistic', 'p-value', 'Lags', '#comments used'])
    for key, value in adf[4].items():
        output["Critical Value (%s)" %key] = value
    
    return print(output)


def kpss_test(dataseries):
    """This checks data stationarity (KPSS Test) and prints the results"""
    
    kpss_input = kpss(dataseries)
    output = pd.Series(kpss_input[0:3], index = ['KPSS Statistic', 'p-value', 'Lags'])
    
    for key, value in kpss_input[3].items():
        output["Critical Value (%s)" %key] = value 
    
    return print(output)


# %%
#User input: Training data

adfuller_test(train)
kpss_test(train)

# %%
def order_parameters(training_data):
    """This performs a grid search for best model parameters"""
    
    search_params = auto_arima(train, start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True, d = 1, D = 1, 
                            trace = True, error_action = 'ignore', suppress_warnings = True, stepwise = True)

    print("AIC = ", round(search_params.aic(), 2))
    print("BIC = ", round(search_params.bic(), 2))

# %%
#User input: Train set

order_parameters(train)

# %%
def train_model(training_data, p, d, q, P, D, Q):
    """This executes model training and prints model summary"""
    
    model = sm.tsa.statespace.SARIMAX(training_data, order = (p, d, q), seasonal_order = (P, D, Q, 7))
    model_fit = model.fit()
    
    return model_fit, print(model_fit.summary().tables[0])


# %%
#User inputs: Training data & parameters found from the search

model_fit, summary = train_model(train, 0, 1, 0, 2, 1, 0)

# %%
def test_model(test_data):
    """This executes model testing"""
    
    output_data = model_fit.get_forecast(steps = len(test_data))
    pred_data = output_data.predicted_mean
    return pred_data


# %%
#User input: Test set

pred = test_model(test)

# %%
def model_output(predicted_data):
    """This plots test and predicted datasets for a visual comparison"""
    
    forecast_data = pd.DataFrame(predicted_data, index = test.index, columns = ['Forecast'])
    output = pd.concat([test, forecast_data], axis = 1)
    
    return output.plot(title = site_id), output


# %%
#User input: Forecasted dataset

plot, resultsdf = model_output(pred)

# %%
def check_error(orig, fore):
    """This function generates performance metrics"""
    
    mse = mean_squared_error(orig, fore)
    rmse = round(sqrt(mse), 2)
    mape = round(np.mean(np.abs((orig - fore) / orig)) * 100, 2)
    metrics = [rmse, mape]
    
    print('RMSE = ', metrics[0])
    print('MAPE = ', metrics[1])
    print("CI = 95%")


def plot_error(output_data):
    """This generates residual graphs"""
    
    ## QQ-plot of the residual
    sm.graphics.qqplot(output_data.iloc[:, 2], line = 'r')
    
    # Autocorrelation plot of the residual
    plot_acf(output_data.iloc[:, 2], lags = len(output_data.iloc[:, 2])-1, zero = False)
       
    return plt.show()

# %%
#Create error column, then plot errors

resultsdf['Error'] = resultsdf['Forecast'] - resultsdf['Energy Consumption (kWh)']
#resultsdf.to_csv('')


check_error(test, pred)
plot_error(resultsdf)


