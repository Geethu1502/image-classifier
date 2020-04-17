#!/usr/bin/env python
# coding: utf-8

# In[9]:


#importing dependancies
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[10]:


# using pandas to read the database stored in the same folder
data=pd.read_csv('mnist_train.csv')


# In[11]:


#viewing coloumn heads
data.head()


# In[17]:


#extracting data from the dataset and viewing them up close
a=data.iloc[4,1:].values


# In[18]:


#reshaping extracted data into a reasonable size
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[19]:


#preparing the data
#seperating labels and data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


# In[20]:


# creating test and train sizes/batches 
x_train,x_test,y_train,y_test=train_test_split(df_x, df_y, test_size=0.2, random_state=4)


# In[21]:


# check data
y_train.head()


# In[22]:


# call rf classifier
rf=RandomForestClassifier(n_estimators=100)


# In[25]:


# fit the model
rf.fit(x_train, y_train)


# In[26]:


# prediction on test data
pred=rf.predict(x_test)


# In[27]:


pred


# In[29]:


# check prediction accuracy
s=y_test.values

#calculate number of correctly predicted values
count=0
for i  in range (len(pred)):
    if pred[i]==s[i]:
        count=count+1


# In[30]:


count


# In[31]:


# total values that the prediction code was run on
len(pred)


# In[33]:


# accuracy value
11608/12000


# In[ ]:




