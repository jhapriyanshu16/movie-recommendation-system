#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Reading data
movies = pd.read_csv('data/movies.csv')
credits = pd.read_csv('data/credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


# marging movies and credits dataframe and updating it in movies
movies = movies.merge(credits,on='title')


# In[6]:


movies.head()


# Data cleaning : 

# In[7]:


#listing all the columns for reference
movies.columns


# In[8]:


#keeping only useful columns for recommendations 
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


movies.shape


# In[11]:


#this gives the count of the null values in the dataframe movies
movies.isnull().sum()


# In[12]:


# There are 3 null values so we will remove those 
movies.dropna(inplace=True)


# In[13]:


movies.shape


# Those three rows with null values are deleted

# In[14]:


# check if movies has duplicated values or not 
movies.duplicated().sum()


# No duplicated values 

# In[15]:


movies.iloc[0]['genres']


# This is string but we need list.

# In[16]:


#converting string into list using ast.literal_eval(). This will give a list of dictonories. Then we will append the value for the key (name) in dictonories
#This function will return value for the key - name (which is genres) 
import ast

def convert(string):
    l = []
    for i in ast.literal_eval(string):
        l.append(i['name'])
    return l


# In[17]:


#updating this list of genres in genres column
movies['genres']= movies['genres'].apply(convert)


# In[18]:


movies.head(2)


# In[19]:


movies.iloc[0]['keywords']


# In[20]:


movies['keywords']= movies['keywords'].apply(convert)


# In[21]:


movies.head(2)


# In[22]:


movies.iloc[0]['cast']


# In[23]:


import ast

def convert_cast(string):
    counter = 0
    l = []
    for i in ast.literal_eval(string):
        if (counter<3):
            l.append(i['name'])
    return l


# In[24]:


#There are a lot of cast but we will append only four cast in our list
movies['cast']= movies['cast'].apply(convert_cast)


# In[25]:


movies.head()


# In[26]:


movies.iloc[0]['crew']


# In[27]:


def fectch_director (text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


# In[28]:


movies['crew']= movies['crew'].apply(fectch_director)


# In[29]:


movies.head()


# In[30]:


#converting overview column into list:
movies['overview']= movies['overview'].apply(lambda x: x.split())


# In[31]:


movies.head()


# In[32]:


#Sam Worthington
#SamWorthington
def remove_space (word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))
    return l


# In[33]:


movies['cast']= movies['cast'].apply(remove_space)
movies['crew']= movies['crew'].apply(remove_space)
movies['keywords']= movies['keywords'].apply(remove_space)
movies['genres']= movies['genres'].apply(remove_space)


# In[34]:


movies.head()


# In[35]:


#concatenating all the columns 
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[36]:


movies.head(5)


# In[37]:


movies.loc[0]['tags']


# In[38]:


# new dataframe with important columns only
new_df = movies[['movie_id','title','tags']]


# In[39]:


new_df.head()


# In[40]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()


# In[41]:


new_df.iloc[0]['tags']


# In[42]:


new_df['tags']= new_df['tags'].apply(lambda x : x.lower())


# In[43]:


new_df.head()


# The Porter stemming algorithm is a process for removing suffixes from words in English :

# In[44]:


import nltk
from nltk.stem import PorterStemmer


# In[45]:


#It removes suffixes
ps = PorterStemmer()


# In[46]:


def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)


# In[47]:


new_df['tags'] = new_df['tags'].apply(stems)


# In[48]:


new_df.iloc[0]['tags']


# Counter Vectorization:
# 
# 

# In[49]:


# It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer (max_features=5000, stop_words= 'english')


# In[50]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[51]:


vector


# In[52]:


from sklearn.metrics.pairwise import cosine_similarity


# In[53]:


similary = cosine_similarity(vector)


# In[54]:


similary


# In[55]:


def recommend (movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted (list(enumerate(similary[index])), reverse=True, key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[62]:


recommend('Spider-Man')


# In[63]:


recommend('Avatar')


# In[64]:




# In[ ]:




