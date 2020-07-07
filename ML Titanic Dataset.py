
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


train_data = pd.read_csv('train.csv')
train_data.head(5)


# In[6]:


test_data = pd.read_csv('test.csv')
test_data.head(5)


# In[17]:


females = train_data.loc[train_data.Sex == "female"]["Survived"]
percentage_females = sum(females)/len(females)*100
print("percentage of woman who survived are: ", round(percentage_females,3), '%')


# In[18]:


males = train_data.loc[train_data.Sex == "male"]["Survived"]
percentage_males = sum(males)/len(males)*100
print("percentage of male who survived are: ", round(percentage_males,3), "%")


# In[1]:


from sklearn.ensemble import RandomForestClassifier
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

classifier = RandomForestClassifier (n_estimators = 100, max_depth = 5, random_state = 1)
classifier.fit(X,y)
predictions = classifier.predict(X_test)

output = pd.DataFrame({'PassengerID':test_data.PassengerId, 'Survived': predictions})
output.to_csv('siddhesh_titanic_submisssion.csv', index = False)
print("CSV successfully generated")

