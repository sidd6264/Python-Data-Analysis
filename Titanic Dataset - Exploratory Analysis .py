
# coding: utf-8

# In[2]:


import pandas as pd
train_file = 'train.csv'

train_data = pd.read_csv(train_file)


# In[3]:


len(train_data)


# In[5]:


train_data.head()


# In[6]:


train_data.count()


# In[7]:


train_data["Age"].min(), train_data["Age"].max()


# In[8]:


train_data["Survived"].value_counts()


# In[9]:


train_data["Sex"].value_counts()


# In[10]:


train_data["Pclass"].value_counts()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')

alpha_color = 0.5
train_data["Survived"].value_counts().plot(kind = 'bar')


# In[13]:


train_data["Sex"].value_counts().plot(kind = 'bar', alpha = alpha_color)


# In[14]:


train_data["Pclass"].value_counts().sort_index().plot(kind = 'bar', alpha = alpha_color)


# In[15]:


train_data.plot(kind = 'scatter', x='Survived', y ='Age')


# In[16]:


train_data[train_data['Survived'] == 1]['Age'].value_counts().sort_index().plot(kind='bar')


# In[5]:


bins = [0,10,20,30,40,50,60,70,80]
train_data['AgeBin'] = pd.cut(train_data['Age'],bins)


# In[7]:


train_data[train_data['Survived'] == 1]['AgeBin'].value_counts().sort_index().plot(kind='bar')


# In[8]:


train_data[train_data['Survived'] == 0]['AgeBin'].value_counts().sort_index().plot(kind='bar')


# In[11]:


train_data['AgeBin'].value_counts().sort_index().plot(kind = 'bar')


# In[17]:


#first class males who survived
train_data[(train_data['Sex'] == 'male') & (train_data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')


# In[18]:


#third class males who survived
train_data[(train_data['Sex'] == 'male') & (train_data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')


# In[19]:


#first class females who survived
train_data[(train_data['Sex'] == 'female') & (train_data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')


# In[20]:


#third class females who survived
train_data[(train_data['Sex'] == 'female') & (train_data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')

