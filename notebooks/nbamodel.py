import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

"""# **Web Scraping Ratings**"""

url = 'https://hoopshype.com/2015/11/05/these-are-the-ratings-of-all-players-in-nba-2k16/'

r = requests.get(url)
data = r.text

soup = BeautifulSoup(data)
soup.prettify

res = requests.get(url)
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0] 

#Ratings by year = rby
rby = pd.read_html(str(table))
print(rby[0].to_json(orient='records'))

rbyTab = tabulate(rby[0], headers='keys', tablefmt='plain') 
rbySplit = rbyTab.split('\n')
print(rbySplit)

data = []
for a in rbySplit:
  a = a.split(' ')
  a[:] = [x for x in a if x != '']
  data.append(a)

data.pop(1)
data[:5]

df = pd.DataFrame(data=data)
df = df.loc[::2]
df.dropna(axis = 0)
df.head(5)

df = df.iloc[1:]
df.drop(df.columns[[-1,]], axis=1, inplace=True)
df.drop(df.columns[[0,]], axis=1, inplace=True)
df.columns = ['First', 'Last', '2K14', '2K15', '2K16']
df.tail(5)

df['Name'] = df['First'].str.cat(df['Last'],sep=" ")
df.sort_values(by=['Name'], inplace=True)
col2 = df.pop("Name")
df.insert(2, col2.name, col2)
df.drop(df.columns[[0,1]], axis = 1, inplace=True)
df.tail(5)

"""# **Joining Ratings and Metrics**"""

#uYY = totals for years YY = 14, 15, 16
u14 = 'https://www.basketball-reference.com/leagues/NBA_2014_totals.html'
u15 = 'https://www.basketball-reference.com/leagues/NBA_2015_totals.html'

#tYY = totals for years YY = 14, 15, 16
t14 = requests.get(u14)
t15 = requests.get(u15)

#dYY = data for years YY = 14, 15, 16
d14 = t14.text
d15 = t15.text

#sYY = parse urls for years YY = 14, 15, 16
s14 = BeautifulSoup(d14)
s14.prettify 

s15 = BeautifulSoup(d15)
s15.prettify 

#rYY = requests.get for years YY = 14, 15, 16
r14 = requests.get(u14)
s14 = BeautifulSoup(r14.content,'lxml')
table14 = s14.find_all('table')[0] 

r15 = requests.get(u15)
s15 = BeautifulSoup(r15.content,'lxml')
table15 = s15.find_all('table')[0] 

#rtYY = ratings for players in years YY = 14, 15, 16
rt14 = pd.read_html(str(table14))
rt15 = pd.read_html(str(table15))

print(rt15[0].to_json(orient='records'))

#rtYYTab and rtYYSplit to tabulate and splot player stats in years YY = 14, 15, 16
rt14Tab = tabulate(rt14[0], headers='keys', tablefmt='plain') 
rt14Split = rt14Tab.split('\n')

rt15Tab = tabulate(rt15[0], headers='keys', tablefmt='plain') 
rt15Split = rt15Tab.split('\n')

print(rt15Split)

#spYY for limiter split data in years YY = 14, 15, 16
sp14 = []
for a in rt14Split:
  a = a.split(' ')
  a[:] = [x for x in a if x != '']
  sp14.append(a)
sp14.pop(1)
sp14[:5]

sp15 = []
for a in rt15Split:
  a = a.split(' ')
  a[:] = [x for x in a if x != '']
  sp15.append(a)
sp15.pop(1)
sp15[:5]

print(sp15)

#dfYY for data frames of player stats in years YY = 14, 15, 16 
df14 = pd.DataFrame(data=sp14)
df14.dropna(axis = 0)
df14header = df14.iloc[0] 
df14 = df14[1:] 
df14.columns = df14header

#Concatenate first and last names
df14['Name'] = df14['Pos'].str.cat(df14['Age'],sep=" ")
col = df14.pop("Name")
df14.insert(2, col.name, col)
df14.sort_values(by=['Pos'], inplace=True)
df14.drop(df14.columns[[0,3,4]], axis = 1, inplace=True)
df14.columns[0:].tolist()

#Fix headers shifted from web scraping via df14.columns[:n].tolist()
List14 = ['Rk', 'Name', 'Pos','Age', 'Tm','G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%',
          'ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS', None, None]
df14.columns = df14.columns[:0].tolist() + List14
df14.drop(df14.columns[[-1, -2]], axis=1, inplace=True)

df14.tail(5)

#dfYY for data frames of player stats in years YY = 14, 15, 16 
df15 = pd.DataFrame(data=sp15)
df15.dropna(axis = 0)
df15header = df15.iloc[0] 
df15 = df15[1:] 
df15.columns = df15header

#Concatenate first and last names
df15['Name'] = df15['Pos'].str.cat(df15['Age'],sep=" ")
col15 = df15.pop("Name")
df15.insert(2, col15.name, col15)
df15.sort_values(by=['Pos'], inplace=True)
df15.drop(df15.columns[[0,3,4]], axis = 1, inplace=True)
df15.columns[0:].tolist()

#Fix headers shifted from web scraping via dfXX.columns[:n].tolist()
List15 = ['Rk', 'Name', 'Pos','Age', 'Tm','G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%',
          'ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS', None, None]
df15.columns = df15.columns[:0].tolist() + List15
df15.drop(df15.columns[[-1, -2]], axis=1, inplace=True)

df15.tail(5)

nba = pd.merge(df14, df, on='Name', sort=False, how='right')
nba = pd.merge(df15, df, on='Name', sort=False, how='right')
nba = nba[nba['Pos'].notna()]
nba.isna().sum(axis = 0)

"""## Type processing and encoding"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

nba['2K14'] = nba['2K14'].apply(pd.to_numeric, args=('coerce',))
nba['2K15'] = nba['2K15'].apply(pd.to_numeric, args=('coerce',))
nba['2K16'] = nba['2K16'].apply(pd.to_numeric, args=('coerce',))
nba.tail(5)

def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True

le.fit_transform(nba['Pos'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

le.fit_transform(nba['Tm'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

nba.tail(5)

df14.iloc[[10]]
nba.iloc[[10]]
df14.tail(5)
nba.tail(5)
df14.describe
df14.Name.nunique()

"""# **Models**

## Preprocessing
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

nba = nba.dropna(subset=['2K14', '2K15'], thresh=1)
nba.isnull().sum(axis = 0)

nbac = nba.drop(['Name'], axis=1)
nbac.dtypes

print(pd.unique(nbac["Age"].values.ravel()))

nbac.columns.tolist()

nbac[['Age', 'G', 'GS', 'MP','FG','FGA','FG%','3P','3PA','2P','2PA','2P%','eFG%', 'FT',
 'FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']] = nbac[['Age', 'G', 'GS', 'MP','FG','FGA','FG%','3P','3PA','2P','2PA','2P%','eFG%', 'FT',
 'FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']].apply(pd.to_numeric, errors='coerce')

nbac = nbac[nbac["Pos"] != 'Jr.']

print(pd.unique(nbac["Pos"].values.ravel()))

nbac.tail(5)

nbac['2K14'].mask(nbac['2K14'].isnull(), nbac['2K15'], inplace=True)
nbac['2K15'].mask(nbac['2K15'].isnull(), nbac['2K14'], inplace=True)
#nbac[['2K14', '2K15', '2K16']] = StandardScaler().fit_transform(nbac[['2K14', '2K15', '2K16']])
nbac.tail(5)

"""# Regression Models"""

nbac = nbac.drop(['3P%', 'FT%'], axis=1)

#X (features) and Y (output/response) 
X = nbac.drop(['2K16', 'Pos', 'Tm'], axis=1)
y = nbac['2K16']

#Splitting 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=811, test_size = 0.2)
X_test.shape

X_train

"""## Linear Regression"""

#Model 1: Linear Regression Model
LinearRegression()

#Fit model
nba_player_model = LinearRegression().fit(X_train, y_train)

#Making predictions!
preds = pd.DataFrame(nba_player_model.predict(X_test)) 
preds1 = pd.DataFrame(nba_player_model.predict(X_train)) 
#print(y_test)

#Combining the data frames with the predictions
numbers = pd.concat([preds1, preds], ignore_index=True)
Machine_Predictions = pd.DataFrame(columns=['2K16_Predictions'])
Machine_Predictions['2K16_Predictions'] = numbers[0]
nbac['2K16_Predictions'] = Machine_Predictions['2K16_Predictions']

print("rms error is: " + str(((nbac['2K16'] - nbac['2K16_Predictions']) ** 2).mean() ** .5))

print('Model score of linear regression on test set: {:.5f}'.format(nba_player_model.score(X_test, y_test)))

"""## Logistic Regression"""

#Model 2: Logistic Regression Model
LogisticRegression()

#Fit model
nba_player_model_log = LogisticRegression().fit(X_train, y_train)

#Making predictions!
preds = pd.DataFrame(nba_player_model_log.predict(X_test)) 
preds1 = pd.DataFrame(nba_player_model_log.predict(X_train)) 
#print(y_test)

#Combining the data frames with the predictions
numbers = pd.concat([preds1, preds], ignore_index=True)
Machine_Predictions = pd.DataFrame(columns=['2K16_Predictions'])
Machine_Predictions['2K16_Predictions'] = numbers[0]
nbac['2K16_Predictions'] = Machine_Predictions['2K16_Predictions']

nbac.head(10)

print("rms error is: " + str(((nbac['2K16'] - nbac['2K16_Predictions']) ** 2).mean() ** .5))

print('Model score of logistic regression on test set: {:.5f}'.format(nba_player_model_log.score(X_test, y_test)))

"""# Clustering Model"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sympy
import seaborn as sns
import matplotlib.ticker as ticker

nbac.select_dtypes(include=np.number).columns.tolist()

#Offensive
max_n_clusters = 10 
iterations = 5000 
category_list = ['Age',
 'G',
 'GS',
 'MP',
 'FG',
 'FGA',
 'FG%',
 '3P',
 '3PA',
 '2P',
 '2PA',
 '2P%',
 'eFG%',
 'FT',
 'FTA',
 'ORB',
 'DRB',
 'TRB',
 'AST',
 'STL',
 'BLK',
 'TOV',
 'PF',
 'PTS',
 '2K14',
 '2K15',
 '2K16']

squared_distance = np.zeros(max_n_clusters)
for k in range(2,max_n_clusters):
    kmeans = KMeans(n_clusters = k, max_iter = iterations).fit(nbac[category_list])
    squared_distance[k] = kmeans.inertia_
    
plt.figure(figsize=(10,10))
plt.plot(squared_distance,c='r')
plt.xlim((2,max_n_clusters))
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average dist to centers')
plt.title('Elbow Plot for K-Means Clustering')
plt.show()

numClusters = 5
km_ = KMeans(n_clusters = numClusters, max_iter = iterations)
km_.fit(nbac[category_list])
km_.labels_

#Cluster assignments
nba_of = nbac
nba_of['Pos'] = km_.labels_

nba_of.groupby(['Pos'], as_index=False).agg('median')

agg = nba_of.groupby(['Pos'], as_index=False).agg('mean')
agg = agg.drop(columns=['Age', '2K16_Predictions', 'eFG%', '2P%', 'FG%', '2K14', '2K15', '2K16'])
agg

#To visualize groups vs. different variable, replace x
fig = plt.figure(figsize=(10,10))
sns.set_context('paper')
nba_of['Pos'] = nba_of['Pos'].astype(str)

km_palette1 = ['r','y','g','b', 'c']

if numClusters == len(km_palette1):
  #TOV, ORB, FT, AST
    ax = sns.scatterplot(x='GS',y='STL',data=nba_of,
                        linewidth=0, alpha=0.6,
                        hue='Pos', palette=km_palette1)
    
    ax.set_xlabel('Games Started')
    ax.set_ylabel('Steals')
else:
    print("Palette is wrong length.")
    
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.title('Games Started vs. Steals');

#To visualize groups vs. different variable, replace x
fig = plt.figure(figsize=(10,10))
sns.set_context('paper')
nba_of['Pos'] = nba_of['Pos'].astype(str)

km_palette1 = ['r','y','g','b', 'c']

if numClusters == len(km_palette1):
  #TOV, ORB, FT, AST
    ax = sns.scatterplot(x='FGA',y='2PA',data=nba_of,
                        linewidth=0, alpha=0.6,
                        hue='Pos', palette=km_palette1)
    
    ax.set_xlabel('Field Goals Attempted')
    ax.set_ylabel('2-Pointers Attempted')
else:
    print("Palette is wrong length.")
    
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.title('Field Goals Attempted vs. 2-Pointers Attempted');

#To visualize groups vs. different variable, replace x
fig = plt.figure(figsize=(10,10))
sns.set_context('paper')
nba_of['Pos'] = nba_of['Pos'].astype(str)

km_palette1 = ['r','y','g','b']

if numClusters == len(km_palette1):
    ax = sns.scatterplot(x='GS',y='FGA',data=nba_of,
                        linewidth=0, alpha=0.6,
                        hue='Pos', palette=km_palette1)
    
    ax.set_xlabel('Field Goals Attempted')
    ax.set_ylabel('2-Pointers Attempted')
else:
    print("Palette is wrong length.")
    
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.title('Games Started vs. Field Goals Attempted');

#To visualize groups vs. different variable, replace x
fig = plt.figure(figsize=(10,10))
sns.set_context('paper')
nba_of['Pos'] = nba_of['Pos'].astype(str)

km_palette1 = ['r','y','g','b', 'c']

if numClusters == len(km_palette1):
    ax = sns.scatterplot(x='ORB',y='DRB',data=nba_of,
                        linewidth=0, alpha=0.6,
                        hue='Pos', palette=km_palette1)
    
    ax.set_xlabel('Offensive Rebounds')
    ax.set_ylabel('Defensive Rebounds')
else:
    print("Palette is wrong length.")
    
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.title('Offensive Rebounds vs. Defensive Rebounds');

#To visualize groups vs. different variable, replace x
fig = plt.figure(figsize=(10,10))
sns.set_context('paper')
nba_of['Pos'] = nba_of['Pos'].astype(str)

km_palette1 = ['r','y','g','b', 'c']

if numClusters == len(km_palette1):
    ax = sns.scatterplot(x='DRB',y='STL',data=nba_of,
                        linewidth=0, alpha=0.6,
                        hue='Pos', palette=km_palette1)
    
    ax.set_xlabel('Offensive Rebounds')
    ax.set_ylabel('Defensive Rebounds')
else:
    print("Palette is wrong length.")
    
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.title('Defensive Rebounds vs. Steals');