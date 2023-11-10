
#!pip install autoviz

from autoviz.AutoViz_Class import AutoViz_Class
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

autoviz = AutoViz_Class().AutoViz('../input/my-datasets/dataset_info.csv')
autoviz

#matplotlib.rc_file_defaults()
ax1 = sns.set_style(style = None, rc = None )

fig, ax1 = plt.subplots(figsize = (22, 10))
b = sns.lineplot(data = df['Usability '], marker = 'o', sort = False, color = 'red', label = "Usability score", ax = ax1)
ax2 = ax1.twinx()
ax1.grid()
ax2.grid()
g = sns.barplot(data = df, x = 'Dataset Title', y = 'Votes', alpha = 0.5, color = 'blue', label = "Votes", ax = ax2)
#b.set_xlabel("Datasets",fontsize = 20)
b.set_ylabel("Usability score", fontsize = 20)
g.set_ylabel("Upvotes", fontsize = 20)
b.tick_params(labelsize = 16)
g.tick_params(labelsize = 16)
b.set_xticklabels(labels = df['Dataset Title'].values.tolist(), rotation = 45)
#g.legend()
#b.legend()

#shuffle dataframe
#from sklearn.utils import shuffle
#df = shuffle(df, random_state = None)
#df.head()

df1 = df.sort_values(by = ['Votes', 'Usability '], ascending = False)
df1.reset_index(inplace = True)
df1.drop('index', axis = 1)

#matplotlib.rc_file_defaults()
ax1 = sns.set_style(style = None, rc = None )

fig, ax1 = plt.subplots(figsize = (22, 10))

b = sns.lineplot(data = df1['Usability '], marker = 'o', color = 'red', label = "Usability", ax = ax1)
ax2 = ax1.twinx()
ax1.grid()
ax2.grid()
g = sns.barplot(data = df1, x = 'Dataset Title', y = 'Votes', alpha = 0.5, color = 'blue', label = "Votes", ax = ax2)
#b.set_xlabel("Datasets",fontsize = 20)
b.set_ylabel("Usability score", fontsize = 20)
g.set_ylabel("Upvotes", fontsize = 20)
b.tick_params(labelsize = 16)
g.tick_params(labelsize = 16)
b.set_xticklabels(labels = df1['Dataset Title'].values.tolist(), rotation = 45)

#Number of datasets by tags
grp = df['Tags'].value_counts().reset_index()
grp.rename(columns = {'Tags':'Dataset count'},inplace = True)
grp.rename(columns = {'index':'Tags'},inplace = True)
grp.style.background_gradient(cmap = 'Reds')

grp1 = df.groupby('Tags').sum().reset_index()
grp2 = grp1.drop('Usability ', axis = 1)
grp2.style.background_gradient(cmap = 'Reds')

