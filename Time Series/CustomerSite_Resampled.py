# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Resampling Hourly Data

# %%
def sample_daily(customer_file):
    """This function down-samples from hourly energy consumption data (csv) to daily energy consumption for all customer-sites over a period"""
    
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
        
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize = (16, 10))
    plt.xlabel('Daily data', fontsize = 12)
    plt.ylabel('Energy Consumption (kWh)', fontsize = 12)
    
    for i in range(0, 1): #len(site_id)
        plt.plot(results['Date'][results['Site'] == site_id[i]], results['Energy Consumption (kWh)'][results['Site'] == site_id[i]], label = site_id[i])
     
    plt.title(customer_name)
    plt.legend()   
    return results, plt.show()


# %%
data_daily, plot = sample_daily('Caliber_kWh_Report.csv')
#data_daily.to_csv('Caliber_Daily.csv')

# %%
def sample_weekly(customer_file):
    """This function down-samples from hourly energy consumption data (csv) to weekly energy consumption for all customer-sites over a period"""
    
    df = pd.read_csv(customer_file, parse_dates = ['Start Time'])
    df.set_index('Start Time', inplace = True)
    df['Date'] = df.index
    customer_name = str(customer_file[:-18])
    
    site_id = df['SiteID'].unique()
    site_name = df['Site Name'].unique()

    resultsdf = pd.DataFrame(columns = ['Site', 'Site Name', 'Energy Consumption (kWh)'])
    results = pd.DataFrame(columns = ['Site', 'Site Name', 'Energy Consumption (kWh)'])
    for i in range(0, len(site_id)):
        
        resultsdf['Energy Consumption (kWh)'] = df['Energy Consumption (kWh)'][df['SiteID'] == site_id[i]].resample('W').sum()
        resultsdf['Site'] = df['SiteID'][df['SiteID'] == site_id[i]].resample('W').first()
        resultsdf['Site Name'] = df['Site Name'][df['Site Name'] == site_name[i]].resample('W').first()
        results = results.append(resultsdf)
        results['Date'] = results.index 
        
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize = (16, 10))
    plt.xlabel('Weekly data', fontsize = 12)
    plt.ylabel('Energy Consumption (kWh)', fontsize = 12)
    
    for i in range(0, len(site_id)):
        plt.plot(results['Date'][results['Site'] == site_id[i]], results['Energy Consumption (kWh)'][results['Site'] == site_id[i]], label = site_id[i])
     
    plt.title(customer_name)
    plt.legend()   
    return results, plt.show()


# %%
data_weekly, plot = sample_weekly('Caliber_kWh_Report.csv')
#data_weekly.to_csv('Caliber_Weekly.csv')


