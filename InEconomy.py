#!/usr/bin/env python
# coding: utf-8

# In[303]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error)
import joblib


# In[304]:


Economy=pd.read_excel("Economy.xlsx")


# In[305]:


Economy


# In[306]:


Economy.shape


# In[307]:


Economy.info()


# In[308]:


Economy.describe()


# In[309]:


Economy.isnull().sum()


# In[310]:


Economy.duplicated().sum()


# In[311]:


Economy.drop_duplicates(inplace=True)


# In[312]:


Economy["Sector"].unique()


# In[313]:


Economy["Sector"].value_counts()


# In[314]:


df=Economy.groupby("Sector")["GDP_Lakh_Crore"].sum()  # Groupby function


# In[315]:


df.plot(kind="pie",autopct="%1.1f%%") 


# In[316]:


Economy["Year"].unique()


# In[317]:


Economy["Year"].value_counts()


# In[318]:


a=Economy.groupby("Year")["GDP_Lakh_Crore"].sum()


# In[319]:


a.plot(kind="bar")


# In[320]:


Economy.groupby(["Year","Quarter"])["GDP_Lakh_Crore"].sum()


# In[321]:


b=Economy.groupby(["Year","Quarter"])["GDP_Lakh_Crore"].sum()


# In[322]:


b.plot(kind="bar")


# In[323]:


Economy.groupby("Sector")["Growth_%"].sum()


# In[324]:


Economy["GNP"]=Economy["GDP_Lakh_Crore"]+Economy["Imports_Crore"]-Economy["Exports_Crore"]


# In[325]:


Economy.head()


# In[326]:


le=LabelEncoder()


# In[327]:


Economy["Quarter_new"]=le.fit_transform(Economy["Quarter"])	


# In[328]:


Economy


# In[329]:


Economy["Sector_new"]=le.fit_transform(Economy["Sector"])	


# In[330]:


Economy


# In[331]:


Economy.drop(columns=["Sector","Quarter"],inplace=True)


# In[332]:


c=MinMaxScaler()


# In[333]:


Economy["GDP_Lakh_Crore"]=c.fit_transform(Economy[["GDP_Lakh_Crore"]])


# In[334]:


Economy


# In[335]:


Economy["Growth_%"]=c.fit_transform(Economy[["Growth_%"]])


# In[336]:


Economy["GNP"]=c.fit_transform(Economy[["GNP"]])


# In[337]:


Economy


# In[338]:


binarizer=Binarizer()


# In[339]:


Economy["GDP_Lakh_Crore_Binary"]=binarizer.fit_transform(Economy[["GDP_Lakh_Crore"]]) 


# In[340]:


Economy


# In[341]:


Economy.drop(columns=["GDP_Lakh_Crore_Binary"],inplace=True)


# In[342]:


Economy


# In[343]:


g=Binarizer(threshold=2020)


# In[344]:


Economy["Year_new"]=g.fit_transform(Economy[["Year"]])


# In[345]:


Economy


# In[346]:


Economy.drop(columns=["Year"],inplace=True)


# In[347]:


Economy


# In[348]:


x =Economy[["Year_new"]]


# In[349]:


y=Economy[["Imports_Crore"]]


# In[350]:


x


# In[351]:


y


# In[352]:


x_train, x_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.2, # 20% for testing
    random_state=42,
    shuffle=True
)


# In[353]:


print(x_train.shape)


# In[354]:


print(x_test.shape)


# In[355]:


print(y_train.shape)


# In[356]:


print(y_test.shape)


# In[357]:


LR=LinearRegression()


# In[358]:


LR.fit(x_train,y_train)


# In[359]:


print("Intercept:", LR.intercept_)
print("Coefficients:",LR.coef_)


# In[360]:


y_pred=LR.predict(x_test)


# In[361]:


y_pred


# In[362]:


mae=mean_absolute_error(y_test,y_pred)


# In[363]:


mae


# In[364]:


mse=mean_squared_error(y_test,y_pred)


# In[365]:


mse



# In[368]:


new_data = pd.DataFrame({ 
"Year_new": [2020]
}) 

prediction = LR.predict(new_data) 
print("Import Value:", prediction[0])


# In[369]:


joblib.dump(LR,"lregression.pkl")


# In[ ]:





# In[ ]:





# In[ ]:




