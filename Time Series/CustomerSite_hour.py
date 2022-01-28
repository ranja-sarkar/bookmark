
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# %%
def read_report(customer_file, site_id):
    """ This function creates univariate time-series chart (energy consumption pattern) for a Customer-SiteID"""
    
    df = pd.read_csv(customer_file)
    df['Date'] = pd.to_datetime(df['Start Time'])
    df = df.set_index(df['Date'])
    
    dff = df.loc[df['SiteID'] == site_id]
    
    x = dff['Date']
    y = dff['Energy Consumption (kWh)']
    data = pd.DataFrame({'Date': x, 'Energy Consumption (kWh)':y})
    
    plt.figure(figsize = (12, 9))
    fig = data['Energy Consumption (kWh)'].plot(title = "Energy Consumption (kWh)", label = site_id)
    plt.grid()
    plt.legend()
    
    return fig

#read_report('JTA_kWh_Report.csv', 'Homerun17')




