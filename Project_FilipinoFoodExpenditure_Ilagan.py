#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datascience import Table
import numpy as np
import matplotlib.pyplot as plots
import scipy as sp
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.formula.api as smf
plots.style.use('fivethirtyeight')
import pandas as pd
import seaborn as sns


# In[2]:


df =  pd.read_csv("Family-Income-and-Expenditure.csv")

df


# In[3]:


df.isnull().sum()


# 

# In[4]:


df = df.drop(["Household Head Occupation","Household Head Class of Worker",],axis=1)

df.info()


# In[5]:


df.shape


# In[6]:


import matplotlib.pyplot as plt
import pandas as pd


# In[7]:


columns_list = df.columns.tolist()

for column in columns_list:
    # Plot the line
    plt.scatter(df[column], df['Total Food Expenditure'])
    plt.xlabel(column)
    plt.ylabel('Total Food Expenditure')
    plt.title(f'Scatter Plot of {column} vs Total Food Expenditure')
    plt.show()


# The data consist of 

#    

# In[8]:


df = Table.from_df(df)


# In[9]:


df.scatter('Total Household Income', 'Total Food Expenditure', fit_line=True)


# In[10]:


df.scatter('Number of Refrigerator/Freezer', 'Total Food Expenditure', fit_line=True)


# In[11]:


df.scatter('Restaurant and hotels Expenditure', 'Total Food Expenditure', fit_line=True)


# In[12]:


df.scatter('Vegetables Expenditure', 'Total Food Expenditure', fit_line=True)


# In[13]:


df.scatter('Total Fish and  marine products Expenditure', 'Total Food Expenditure', fit_line=True)


# In[14]:


df.scatter('Meat Expenditure', 'Total Food Expenditure', fit_line=True)


# In[ ]:





# In[15]:


mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Restaurant and hotels Expenditure")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[16]:


mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Vegetables Expenditure")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[17]:


mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Total Household Income")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[18]:


mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Total Fish and  marine products Expenditure")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[19]:



mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Meat Expenditure")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[20]:


mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Number of Refrigerator/Freezer")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[ ]:





# In[22]:


mod = smf.ols(formula='Q("Total Food Expenditure") ~ Q("Total Household Income")+Q("Restaurant and hotels Expenditure")+Q("Meat Expenditure")+Q("Total Fish and  marine products Expenditure")+Q("Vegetables Expenditure")', data=df.to_df())
res = mod.fit()
print(res.summary())


# In[23]:


import matplotlib.pyplot as plt

# Coefficients and their corresponding independent variables
independent_variables = [
    "Total Household Income",
    "Restaurant and hotels Expenditure",
    "Meat Expenditure",
    "Total Fish and marine products Expenditure",
    "Vegetables Expenditure"
]
coefficients = [0.0149, 1.0079, 1.6481, 1.6227, 2.4375]  # Example coefficients

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.barh(independent_variables, coefficients, color='skyblue')
plt.xlabel('Estimated Change in Total Food Expenditure')
plt.title('Estimated Change in Total Food Expenditure for Each Independent Variable')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest coefficient at the top
plt.show()


# In[ ]:




