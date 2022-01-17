#!/usr/bin/env python
# coding: utf-8

# In[183]:


import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
data = sns.load_dataset('diamonds')


# In[184]:


# Dropping NaN columns
data = data.dropna()
data.head()
# data['table']


# In[137]:


sns.boxplot(x=data['table'])
sns.boxplot(x=data['carat'])
sns.boxplot(x=data['depth'])
sns.boxplot(x=data['price'])
sns.boxplot(x=data['x'])
sns.boxplot(x=data['y'])
sns.boxplot(x=data['z'])


# In[168]:


Q1 = data['table'].quantile(0.25)
Q3 = data['table'].quantile(0.75)

IQR = Q3 - Q1

print(IQR)


# In[169]:


# Locating outiers using IQR

lower_lim = Q1 - (1.5 * IQR)
upper_lim = Q3 + (1.5 * IQR)

outliers_low = (data['table'] < lower_lim)
outliers_up = (data['table'] > upper_lim)

data['table'][(outliers_low | outliers_up)]
# print(data)


# In[170]:


# Removing outliers using IQR

data['table'] = data['table'][~(outliers_low | outliers_up)]
print(data)


# In[171]:


# After removing the outliers

print(data)
sns.boxplot(x = data['table'])
plt.show()


# In[185]:


# Removing outliers using Statistics

from numpy import mean, std

# Calculate statistics
data_mean, data_std = mean(data['depth']), std(data['depth'])

# Identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
print(lower)
print(upper)


# In[192]:


# outliers = [x for x in data['depth'] if x < lower or x > upper]
# print(outliers)

low = data['depth'] > lower
# print(low)
high = data['depth'] < upper

# # Removing outliers

data['depth'] = data['depth'][low & high] 
# print(data)
data = data.dropna()
print(data)


# In[194]:


#After removing the outliers

print(data)
sns.boxplot(x = data['depth'], showfliers=False)
plt.show()


# In[202]:


# Removing outliers using Log Trnasformation

data['price'].head()
sns.boxplot(x = data['price'])


# In[203]:


import numpy as np

data_log = np.log(data['price'])
# print(data['price'].head())

print(data_log.head())


# In[204]:


sns.boxplot(x = data_log)


# In[ ]:




