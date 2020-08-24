#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[14]:


import evaluate


# In[6]:


validation_behaviors = pd.read_csv('/Users/sumitsidana/Downloads/MIND_news_recommendation_dataset/MINDlarge_dev/behaviors.tsv', delimiter = "\t", names = ['impression_id', 'user_id', 'time', 'history', 'impressions'])


# In[7]:


validation_behaviors.head()


# In[9]:


v_b = validation_behaviors[['impression_id', 'impressions']]


# In[10]:


v_b.head()


# In[11]:


def return_index(lst):
    index_str = '['
    for elem in lst[:-1]:
        index_str = index_str + str(lst.index(elem) + 1)+ ','
    index_str = index_str + str(lst.index(lst[-1])+1) + ']'
    return index_str


# In[12]:


v_b['predictions'] = v_b['impressions'].str.split(' ')


# In[13]:


v_b['indexes'] = v_b['predictions'].apply(lambda x: return_index(x))


# In[24]:


def get_labels(lst):
    labels_list = '['
    
    for elem in lst[:-1]:
        label = float(elem.split('-')[1])
        labels_list = labels_list + str(label) +","
    last_label = float(lst[-1].split('-')[1])
    labels_list = labels_list + str(last_label)+']'
    return labels_list


# In[25]:


v_b['labels'] = v_b['predictions'].apply(lambda x: get_labels(x))


# In[26]:


v_b[['impression_id', 'indexes']].to_csv('res/prediction.txt', index = False, header = None, sep = ' ')


# In[27]:


v_b[['impression_id', 'labels']].to_csv('ref/truth.txt', index = False, header = None, sep = ' ')


# In[36]:


get_ipython().run_line_magic('run', "-i 'evaluate.py' './' './'")


# In[31]:


import ast
type(ast.literal_eval(v_b.loc[0].labels)[0])

