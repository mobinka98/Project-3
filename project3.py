#!/usr/bin/env python
# coding: utf-8

# # PROJECT 3

# ###### needed libaries

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# ###### Load the dataset

# In[3]:

import pandas as pd

data = pd.read_csv('dataset.csv')


# In[4]:


data


# ###### Display the data to confirm it's loaded correctly
# 
# 

# In[14]:


print(data.head())


# ###### Assuming the dataset has two columns: 'x' and 'y'

# In[6]:


X = data[['x']]  
y = data['y']    


# ###### Split the data into training and testing sets

# In[7]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ###### Create and train the linear regression model

# In[9]:

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# #####
# # Use the trained model to make predictions on the test data

# In[10]:


y_pred = model.predict(X_test)


# ###### Assess the model's performance using mean squared error
# 
# 

# In[12]:

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# ###### Display the coefficients
# 
# 

# In[13]:


print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

