#!/usr/bin/env python
# coding: utf-8

# # Microchannel heat sink optimization using regression models
# 
# 

# Firstly, nearly 200 data were generated with COMSOL Multiphysics. this problem, we have five major features($N$,$H_R$,$V_R$,$L_R$,$W_R$) and one target ($R_{Th}$).

# In[116]:


# Import lib.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
plt.rcParams["font.family"] = "Times New Roman"
from PIL import Image
from io import BytesIO


# In[117]:


# Reading data
data = pd.read_csv ("E:\machine learning/test-c.csv")
data


# In[118]:


# Generating dataframe
df = pd.DataFrame (data,columns=['N','Rh','Rv','Rl','Rw','Rth'])
df


# In[119]:


df.describe()


# In[120]:


x = pd.DataFrame (df, columns=['N','Rh','Rv','Rl','Rw'])
y = df['Rth'].values.reshape(-1,1)


# In[121]:


# 80% for training and 20% for the test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[122]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[123]:


y_pred = regressor.predict(x_test)


# In[124]:


# Evaluation of the model based on error criteria
print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared Error : ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean squared Error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 score : ", metrics.r2_score(y_test, y_pred))


# Trying polynomial regression model by adding the square of the parameters as new features and adding them to the dataset.

# In[125]:


N2 = df.N**2
Rh2 = df.Rh**2
Rv2 = df.Rv**2
Rl2 = df.Rl**4
Rw2 = df.Rw**2


# In[126]:


df.insert(5, "N2", N2)
df.insert(6, "Rh2", Rh2)
df.insert(7, "Rv2", Rv2)
df.insert(8, "Rl2", Rl2)
df.insert(9, "Rw2", Rw2)
df


# In[127]:


x = pd.DataFrame (df, columns=['N','N2','Rh','Rh2','Rv','Rv2','Rl','Rl2','Rw','Rw2'])
y = df['Rth'].values.reshape(-1,1)


# In[128]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[129]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[130]:


y_pred = regressor.predict(x_test)


# In[131]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared Error : ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean squared Error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 score : ", metrics.r2_score(y_test, y_pred))


# Now we have an accurate model!

# It is the time to visualize the predicted and actual data of the testing proportion.

# In[132]:


Compare = pd.DataFrame ({'Actual':y_test.flatten(),'predict':y_pred.flatten()})
plt.figure(figsize = (8,6), dpi = 300)
plt.plot(Compare,label={'Actual',"predict"})
plt.xlabel ("data index")
plt.ylabel ("$R_{th}$ [K$W^{-1}$]")
plt.title ("Comparison of predicted and actual thermal resistence at Pumping power of 0.05W ")
plt.legend(loc='best')
plt.grid()
plt.show()


# In[133]:


T=np.array(y_test.flatten())


# In[134]:


P=np.array(y_pred.flatten())


# In[135]:


# Import libs.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
plt.rcParams["font.family"] = "Times New Roman"
from PIL import Image
from io import BytesIO


# In[136]:


fig, ax = plt.subplots(figsize=[4, 4], dpi = 300)
ax.annotate('-2%', xy = (0.15, 0.145),fontsize=10)
ax.annotate('+2%', xy = (0.1435, 0.15),fontsize=10)


plt.scatter(T,P,s=12,color = 'b')


x1 = [0,1,2,3]
y1 = [0,1.02,2.04,3.06]
plt.plot(x1, y1,color = '#F97306')
x2 = [0,1,2,3]
y2 = [0,0.979,1.958,2.937]
plt.plot(x2, y2,color = '#F97306')

plt.xlabel ("Actual value of $\mathdefault{R_{th}}$ [$\mathdefault{KW^{-1}}$]",fontsize=10)
plt.ylabel ("Predicted value of $\mathdefault{R_{th}}$ [$\mathdefault{KW^{-1}}$]",fontsize=10)
plt.rcParams.update({'font.size': 8})
handles, labels = plt.gca().get_legend_handles_labels()
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.xlim(0.134,0.16)
plt.ylim(0.134,0.16)

plt.grid(b=True, which='major', color='#DDDDDD', linestyle='-', linewidth=0.8)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#EEEEEE', linestyle='-', linewidth=0.5)
plt.rc('axes', axisbelow=True)
plt.show()


# In[137]:


# Finding the coefficients of the polynomial model
print(regressor.intercept_)
print(regressor.coef_)


# Checking the five external data prediction, while the total original dataset is used for training

# In[138]:


# Reading data
data2 = pd.read_csv ("E:\machine learning/test-cf.csv")
data2


# In[139]:


df2 = pd.DataFrame (data2,columns=['N','Rh','Rv','Rl','Rw'])
df2


# In[140]:


N2 = df2.N**2
Rh2 = df2.Rh**2
Rv2 = df2.Rv**2
Rl2 = df2.Rl**4
Rw2 = df2.Rw**2


# In[141]:


df2.insert(5, "N2", N2)
df2.insert(6, "Rh2", Rh2)
df2.insert(7, "Rv2", Rv2)
df2.insert(8, "Rl2", Rl2)
df2.insert(9, "Rw2", Rw2)
df2


# In[142]:


# Adding the additional data for prediction to the dataset
df3 = df.append(df2)
df3


# In[143]:


train = df3.iloc[:193]
test = df3.iloc[193:]


# In[144]:


test


# In[152]:


# Training with 100% of the original dataset, and allocate the five added data for testing and prediction
x_train = df3[['N','N2','Rh','Rh2','Rv','Rv2','Rl','Rl2','Rw','Rw2']][:193]
y_train = df3["Rth"][:193].values.reshape(-1,1)


# In[153]:


x_test = df3[['N','N2','Rh','Rh2','Rv','Rv2','Rl','Rl2','Rw','Rw2']][193:]


# In[154]:


regressor.fit(x_train, y_train)


# In[155]:


# Prediction of the model for these five cases
y_pred = regressor.predict(x_test)
print(y_pred)


# In[ ]:




