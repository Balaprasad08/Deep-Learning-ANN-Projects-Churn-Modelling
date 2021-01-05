#!/usr/bin/env python
# coding: utf-8

# ## Artificial Neural Network

# In[1]:


# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\prasad\\practice\\Deep Learning-29-12-20\\Churn Modelling')


# In[3]:


# Importing the dataset
df=pd.read_csv('Churn_Modelling.csv')


# In[4]:


df.head(2)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


cat_var=df.select_dtypes(include=['object'])


# In[9]:


cat_var.head(2)


# In[10]:


df.head(2)


# In[11]:


df.Surname.unique()


# In[12]:


X=df.iloc[:,3:13]
X.head(2)


# In[13]:


y=df.iloc[:,-1:]
y.head(2)


# In[14]:


#Create dummy variables
geography=pd.get_dummies(df['Geography'],drop_first=True)
gender=pd.get_dummies(df['Gender'],drop_first=True)


# In[15]:


geography.head(2)


# In[16]:


gender.head(2)


# In[17]:


## Concatenate the Data Frames
X=pd.concat([X,geography,gender],axis=1)


# In[18]:


X.head(2)


# In[19]:


## Drop Unnecessary columns
X.drop(['Geography','Gender'],axis=1,inplace=True)


# In[20]:


X.head(2)


# In[21]:


X.shape


# ### Splitting the dataset into the Training set and Test set

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[24]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Feature Scaling

# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


sc=StandardScaler()


# In[27]:


X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.fit_transform(X_test)


# In[28]:


X_train_sc.shape,X_test_sc.shape


# In[29]:


X_train=pd.DataFrame(X_train_sc,columns=X_train.columns)
X_test=pd.DataFrame(X_test_sc,columns=X_test.columns)


# In[30]:


X_train.head(2)


# In[31]:


X_test.head(2)


# ### Part 2 - Now let's make the ANN!

# In[32]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,ELU,PReLU
from keras.layers import Dropout


# In[33]:


# Initialising the ANN
model=Sequential()


# In[34]:



# Adding the input layer and the first hidden layer
model.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu',input_dim = 11))
#Adding Droupout layers
model.add(Dropout(0.2))

# Adding the Second hidden layer
model.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.3))

# Adding the Third hidden layer
model.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu'))
model.add(Dropout(0.2))

# Adding the Output layer
model.add(Dense(units=1, kernel_initializer='glorot_uniform',activation='sigmoid'))


# In[35]:


# Compiling the ANN
model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])


# In[36]:


# Fitting the ANN to the Training set
model_history=model.fit(
    X_train,y_train,
     batch_size=10,
    epochs=100,
    validation_split=0.33)


# ### Part 3 - Making the predictions and evaluating the model

# In[37]:


# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


# In[38]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[39]:


cm


# In[40]:


# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# In[41]:


score


# ### Save Model in h5

# In[42]:


model.save('churn.h5')


# ### Load h5 Model 

# In[44]:


from keras.models import load_model


# In[45]:


final_model=load_model('churn.h5')


# In[46]:


y_pred_final=final_model.predict(X_test)


# In[47]:


y_pred_final=y_pred_final>0.5


# In[48]:


cm_final=confusion_matrix(y_test,y_pred_final)


# In[49]:


cm_final


# In[50]:


accuracy_score(y_test,y_pred_final)


# In[52]:


final_model.summary()


# In[53]:


final_model.get_weights()


# In[55]:


final_model.optimizer


# In[ ]:




