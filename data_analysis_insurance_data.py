#!/usr/bin/env python
# coding: utf-8

# # Insurance data analysis
# # by VINILA GUMMADI

# In[1]:


#importing all the required libraries for importing data ,for EDA,for modelling,model evaluation and for model validation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


# In[2]:


filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'


# In[3]:


df = pd.read_csv(filepath)


# In[4]:


df.head()


# In[5]:


headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers


# In[6]:


df.head()


# In[7]:


# replacing ? entries with NaN 
df.replace('?', np.nan, inplace = True)


# In[8]:


print(df.info())


# In[9]:


# replacing missing values of smoker value with most frequently appearing value as its a categorial value
is_smoker = df['smoker'].value_counts().idxmax()
df["smoker"].replace(np.nan, is_smoker, inplace=True)

# replacing missing values of age with mean of age
mean_age = df['age'].astype('float').mean(axis=0)
df["age"].replace(np.nan, mean_age, inplace=True)

# Updating data types
df[["age","smoker"]] = df[["age","smoker"]].astype("int")

print(df.info())


# In[10]:


df[["charges"]] = np.round(df[["charges"]],2)
print(df.head())


# In[11]:


sns.regplot(x="bmi", y="charges", data=df, line_kws={"color": "red"})
plt.ylim(0,)


# In[12]:


sns.boxplot(x="smoker", y="charges", data=df)


# In[13]:


print(df.corr())


# In[14]:


# constructing a model
X = df[['smoker']]
Y = df['charges']
lm = LinearRegression()
lm.fit(X,Y)
print(lm.score(X, Y))


# In[15]:


# for improvement of model with all attributes available
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lm.fit(Z,Y)
print(lm.score(Z, Y))


# In[16]:


# by normalizing and using higher order polynomial we are grabing a more good fit for the data eventually increasing the r2score
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(r2_score(Y,ypipe))


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)


# In[18]:


RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))


# In[19]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))

