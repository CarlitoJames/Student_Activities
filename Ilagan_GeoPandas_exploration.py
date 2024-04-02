#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install geopandas


# In[2]:


import sys
sys.path.append(r"C:\Users\cjame\Documents\college\2ND YEAR 3RD TERM\TIme series")

from figure_labeler import *
from IPython.display import HTML
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# We can see different GeoJSON levels from 0-3. Below are their breakdowns.
# <br>
# - **Level 0**: This typically represents the entire country as a single polygon. It's the highest level of administrative division, encompassing the entire territory of the Philippines.
# 
# - **Level 1**: This represents the first-level administrative divisions, which in the case of the Philippines are the regions. The Philippines is divided into several regions, each consisting of multiple provinces.
# 
# - **Level 2**: This represents the second-level administrative divisions, which are the provinces. Each region in the Philippines is divided into provinces.
# 
# - **Level 3**: This represents the third-level administrative divisions, which are the municipalities or cities (depending on the classification). Provinces in the Philippines are further divided into municipalities and cities.

# In[3]:


fl = FigureLabeler();
pd.options.mode.chained_assignment = None


# In[4]:


gdf_0 = gpd.read_file("./Data/level_0/gadm41_PHL_0.json")


# In[5]:


gdf_0


# | Column    | Description                               |
# |-----------|-------------------------------------------|
# | GID_0     | Unique identifier for the country at Level 0 |
# | COUNTRY   | Name of the country (Philippines)            |
# | geometry  | Geometric data describing the country's shape  |
# 
# Here is the data dictionary of the level 0 gadm file that we read from json.

# In[6]:


gdf_1 = gpd.read_file("./Data/level_1/gadm41_PHL_1.json")


# In[7]:


fl.table_caption("level 1 Geojson data", "data in level 1 GeoJson")
gdf_1


# In[8]:


fl.fig_caption("Philippines Map","oultine of Philippines Map")
gdf_0.boundary.plot(figsize=(8, 8))
plt.show()


# In[9]:


fl.fig_caption("Philippines Map",
               "my prefered color of the map")
gdf_0.boundary.plot(figsize=(8, 8), color='darkgoldenrod')
plt.xticks([])
plt.yticks([])


# In[10]:


fl.fig_caption("Philippines Map",
               "oultine of Philippines Map and it's regions")
gdf_1.boundary.plot(figsize=(8, 8), color='darkgoldenrod')
plt.xticks([])
plt.yticks([])


# In[11]:


batangas = gdf_1[gdf_1["NAME_1"]== "Batangas"]


# In[12]:


fl.fig_caption("Batangas Map",
               "oultine of Batangas Map")
batangas.boundary.plot(figsize=(8, 8), color='darkgoldenrod')
plt.xticks([])
plt.yticks([])


# In[13]:


gdf_3 = gpd.read_file("./Data/level_3/gadm41_PHL_3.json")


# In[14]:


batangas = gdf_3[gdf_3["NAME_1"]== "Batangas"]
batangas.boundary.plot(figsize=(8, 8), color='darkgoldenrod')
plt.xticks([])
plt.yticks([])
fl.fig_caption("Batangas Map",
               "oultine of Batangas Map with its municipalities")


# In[15]:


lobo_batangas = batangas[batangas["NAME_2"] == "Lobo"]

# Plot the boundary for Lobo in Batangas
batangas.boundary.plot(figsize=(8, 8), color='lightgrey')

# Plot the boundary for Lobo in Batangas
lobo_batangas.boundary.plot(ax=plt.gca(), color='darkgoldenrod')

# Remove ticks
plt.xticks([])
plt.yticks([])
fl.fig_caption("Batangas Map",
               "Highlighting Lobo map within Batangas Map")


# In[16]:


lobo_batangas.boundary.plot(figsize=(10, 10), color='darkgoldenrod')
plt.xticks([])
plt.yticks([])
fl.fig_caption("Lobo Map",
               "oultine of Lobo Map and its municipalities")


# In[17]:


fl.table_caption("Finpop dataset", "different finance of the Philippines")
df_finpop = pd.read_csv("./Data/financial_pop.csv")
df_finpop.head()


# | Column                       | Description                                              |
# |------------------------------|----------------------------------------------------------|
# | pop                          | Population of the municipality/province                  |
# | tot_local_sources            | Total revenue from local sources                         |
# | tot_tax_revenue              | Total tax revenue collected                              |
# | tot_current_oper_income      | Total current operating income                           |
# | total_oper_expenses          | Total operating expenses                                 |
# | net_oper_income              | Net operating income (total current operating income - total operating expenses) |
# | total_non_income_receipts    | Total non-income receipts                                |
# | capital_expenditure          | Capital expenditure (investment in assets)               |
# | total_non_oper_expenditures | Total non-operating expenditures                         |
# | cash_balance_end             | Cash balance at the end of the period                    |
# | shp_province                 | Name of the province                                     |
# | shp_municipality             | Name of the municipality                                 |
# 

# In[18]:


df_finpop_batangas = df_finpop[df_finpop['shp_province'] == 'Batangas']


# In[19]:


fl.table_caption("Finpop dataset", "Batangas specific dataset")
df_finpop_batangas.head()


# In[20]:


bat_population = df_finpop_batangas.sort_values(by="pop", ascending=False)


# In[21]:


bat_population.set_index('shp_municipality', inplace=True)


# In[22]:


fl.fig_caption("Batangas Population",
               "Batangas municipalities with their respective Population descending")
plt.figure(figsize=(10,10))
bat_population['pop'].plot(kind='barh',color ="gray")
lobo_index = bat_population.index.get_loc('Lobo')
plt.barh(lobo_index, bat_population.loc['Lobo', 'pop'], color='darkgoldenrod')
plt.title("Municipalities with their respective Population")
plt.xlabel('Population')
plt.ylabel('Municipality')
plt.gca().invert_yaxis()
plt.show()


# In[23]:


bat_tax_revenue = df_finpop_batangas.sort_values(by="tot_tax_revenue", ascending=False)
bat_tax_revenue.set_index('shp_municipality', inplace=True)


# In[24]:


fl.fig_caption("Batangas total tax revenue",
               "Batangas municipalities with their respective total tac revenue descending")
plt.figure(figsize=(10,10))
bat_tax_revenue['tot_tax_revenue'].plot(kind='barh',color ="gray")
lobo_index = bat_tax_revenue.index.get_loc('Lobo')
plt.barh(lobo_index, bat_tax_revenue.loc['Lobo', 'tot_tax_revenue'], color='darkgoldenrod')
plt.title("Municipalities with their respective  total tax revenue")
plt.xlabel('capital expenses')
plt.ylabel('Municipality')
plt.gca().invert_yaxis()
plt.show()


# In[25]:


batangas = gdf_3[gdf_3["NAME_1"]== "Batangas"]
batangas.head()


# In[26]:


batangas.shape


# In[27]:


merge_batangas = batangas.merge(df_finpop_batangas, left_on = ['NAME_2'], right_on = ['shp_municipality'])


# In[28]:


fl.table_caption("Merge batangas dataset", "finpop dataset and GeoJson dataset merge")
merge_batangas


# In[ ]:





# In[29]:


fl.fig_caption("Batangas total tax revenue in heatmap",
               "Batangas municipalities with their respective total tax revenue in heat map but uncleaned")
ax = merge_batangas.plot(column='tot_tax_revenue',
             figsize=(20,20),
             cmap='OrRd',
             legend=True,
             edgecolor='black',
             linewidth=0.5);


# In[30]:


batangas_cleaned = batangas[batangas['NAME_2'] != 'Taallake']


# In[31]:


df_finpop_batnagas_cleaned = df_finpop_batangas.copy()
df_finpop_batnagas_cleaned.reset_index(inplace=True)


# In[32]:


batangas_cleaned1 = batangas.replace('Taallake','Taal')


# In[33]:


df_finpop_batnagas_cleaned1 = df_finpop_batangas.copy()
df_finpop_batnagas_cleaned1.reset_index(inplace=True)


# In[34]:


df_finpop_batnagas_cleaned['shp_municipality'] = df_finpop_batnagas_cleaned['shp_municipality'].str.replace(' ', '')


# In[35]:


merge_batangas_cleaned =  batangas_cleaned.merge(df_finpop_batnagas_cleaned, left_on=['NAME_2'], right_on=['shp_municipality'])


# In[36]:


merge_batangas_cleaned['NAME_2'].shape


# In[37]:


fl.fig_caption("Batangas total tax revenue in heatmap |Type 1|",
               "Cleaned heat map tax revenue but taal lake is remove in the heat map")

ax = merge_batangas_cleaned.plot(column='tot_tax_revenue',
             figsize=(20,20),
             cmap='OrRd',
             legend=True,
             edgecolor='black',
             linewidth=0.5);


# In[38]:


merge_batangas_cleaned1 =  batangas_cleaned1.merge(df_finpop_batnagas_cleaned, left_on=['NAME_2'], right_on=['shp_municipality'])


# In[39]:


merge_batangas_cleaned1['NAME_2'].shape


# In[40]:


fl.fig_caption("Batangas total tax revenue in heatmap |Type 2|",
               "Cleaned heat map tax revenue but Taal Lake is combined with Taal tax revenue as Taal Lake doesn't have one")
ax = merge_batangas_cleaned1.plot(column='tot_tax_revenue',
             figsize=(20,20),
             cmap='OrRd',
             legend=True,
             edgecolor='black',
             linewidth=0.5);


# In[41]:


fl.fig_caption("Batangas total tax revenue in heatmap Lobo Highlighted",
               "Cleaned heat map tax revenue but Taal Lake is combined with Taal tax revenue as Taal Lake doesn't have one")
ax = merge_batangas_cleaned1.plot(column='tot_tax_revenue',
                                  figsize=(20, 20),
                                  cmap='OrRd',
                                  legend=True,
                                  edgecolor='black',
                                  linewidth=0.5)
lobo_boundary = merge_batangas_cleaned1[merge_batangas_cleaned1['NAME_2'] == 'Lobo'].boundary
lobo_boundary.plot(ax=ax, color='black', linewidth=2)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[42]:


fl.fig_caption("Batangas total Population with Lobo Highlighted",
               "Type 2 cleaned heat map, Lobo highlighted")
ax = merge_batangas_cleaned1.plot(column='pop',
                                  figsize=(20, 20),
                                  cmap='YlOrBr',
                                  legend=True,
                                  edgecolor='black',
                                  linewidth=0.5)
lobo_boundary = merge_batangas_cleaned1[merge_batangas_cleaned1['NAME_2'] == 'Lobo'].boundary
lobo_boundary.plot(ax=ax, color='black', linewidth=2)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[ ]:





# In[ ]:




