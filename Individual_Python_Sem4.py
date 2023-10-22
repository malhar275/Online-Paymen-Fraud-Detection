#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("Online_Payment_Dataset.csv")
print(data.head())


# In[2]:


# step: represents a unit of time where 1 step equals 1 hour
# type: type of online transaction
# amount: the amount of the transaction
# nameOrig: customer starting the transaction
# oldbalanceOrg: balance before the transaction
# newbalanceOrig: balance after the transaction
# nameDest: recipient of the transaction
# oldbalanceDest: initial balance of recipient before the transaction
# newbalanceDest: the new balance of recipient after the transaction
# isFraud: fraud transaction


# In[3]:


# to check null values are there or not
print(data.isnull().sum())


# In[4]:


# Exploring transaction type
print(data.type.value_counts())


# In[14]:


# Sort the DataFrame by the step column
data = data.sort_values(by=['step'], ascending=True)

# Extract the step and amount columns
step_column = data['step']
amount_column = data['amount']

# Create the line chart
plt.plot(step_column, amount_column)

# Set the title and axis labels
plt.title('Line Chart of Transaction Amounts Over Time')
plt.xlabel('Step')
plt.ylabel('Transaction Amount')

# Show the line chart
plt.show()


# In[5]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values

color_discrete_map = {
    "CASH_OUT": "blue",
    "PAYMENT": "red",
    "CASH_IN":"yellow",
    "TRANSFER": "green",
    "DEBIT":"purple",
}

import plotly.express as px
figure = px.pie(data, values=quantity,names=transactions,height=600,width=800,hole = 0.5,title="Distribution of Transaction Type",
                color_discrete_map=color_discrete_map,)
figure.show()


# In[11]:



import matplotlib.pyplot as plt

plt.scatter(data['step'], data['amount'], marker='o', color='blue', alpha=0.5)
plt.xlabel('Step')
plt.ylabel('Amount')
plt.title('Scatter plot of amount and step')
plt.legend(['Amount'])
plt.show()


# In[12]:


import seaborn as sns
# Create the violin plot
sns.violinplot(x = "type", y = "amount", data=data)
plt.xlabel('Merchant type')
plt.ylabel('Transaction amount')
plt.title('Violin plot of transaction amounts by merchant type')
plt.show()


# In[6]:


# Checking correlation
correlation = data.corr()
print(correlation)


# In[7]:


correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))


# In[8]:


# Now let’s transform the categorical features into numerical. Here I will also transform the values of the isFraud 
# column into No Fraud and Fraud labels to have a better understanding of the output
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4,"DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


# In[9]:


# Now let’s train a classification model to classify fraud and non-fraud transactions. 
# Before training the model, I will split the data into training and test sets
# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[10]:


# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[16]:


data.head()


# In[17]:


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[2, 9839.64, 170136.00, 160296.36]])
print(model.predict(features))


# In[18]:


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))

