from subprocess import check_output

import inline as inline
import matplotlib

print(check_output(["ls", "./"]).decode("utf8"))

import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# %matplotlib inline

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

# plt.show()

data = pd.read_csv("kc_house_data_2.csv")
data.head()
data.describe(include=[np.number])

data.isnull().sum()  # Data not having any NaNs

names = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
         'grade', 'sqft_above', 'sqft_basement', 'zipcode', 'lat', 'long']
df = data[names]
correlations = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 15, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
# plt.show()

data['waterfront'] = data['waterfront'].astype('category', ordered=True)
data['view'] = data['view'].astype('category', ordered=True)
data['condition'] = data['condition'].astype('category', ordered=True)
data['grade'] = data['grade'].astype('category', ordered=False)
data['zipcode'] = data['zipcode'].astype('category', ordered=False)

print(data.dtypes)

# sns.set_style()
sns.regplot(x='sqft_living', y='price', data=data)

sns.regplot(x='sqft_basement', y='price', data=data)

sns.regplot(x='sqft_above', y='price', data=data)

sns.stripplot(x='bedrooms', y='price', data=data)

sns.stripplot(x='bathrooms', y='price', data=data, size=5)

sns.stripplot(x='grade', y='price', data=data, size=5)

data = data[data['bedrooms'] < 10]

data = data[data['bathrooms'] < 8]

print(data.head())

c = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_above', 'grade']
df = data[c]

df = pd.get_dummies(df, columns=['grade'], drop_first=True)

y = data['price']

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=42)

print(x_train.head())

reg = LinearRegression()

reg.fit(x_train, y_train)

print('Coefficients: \n', reg.coef_)

print(metrics.mean_squared_error(y_test, reg.predict(x_test)))

reg.score(x_test, y_test)
# 0.6035

df = pd.get_dummies(data, columns=['waterfront', 'view', 'condition', 'grade', 'zipcode'], drop_first=True)

y = data['price']
df = df.drop(['date', 'id', 'price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=42)

reg.fit(x_train, y_train)

print('Coefficients: \n', reg.coef_)
print(metrics.mean_squared_error(y_test, reg.predict(x_test)))
print(reg.score(x_test, y_test))


