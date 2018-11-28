
# coding: utf-8

# In[1]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')
#Language Classification
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
stopwords.fileids()
used_languages = ['english','portuguese','french','italian','spanish','german']
def detect_language(text,used_languages):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.
    
    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.
    """

    ratios = _calculate_languages_ratios(text,used_languages)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language

def _calculate_languages_ratios(text,used_languages):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}
    

    """

    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        if language in used_languages: 
            stopwords_set = set(stopwords.words(language))
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)

            languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios

get_ipython().system('jupyter nbconvert --to script Language_Recognition.ipynb')

import json

with open('config.json', 'r') as f:
    config = json.load(f)


root_dir = config['FOLDERS']['ROOT_DIR']
path = config['FOLDERS']['PATH']
version = config['VERSION']
results_path = config['FOLDERS']['RESULTS']
version = config['VERSION']
members_list = config['FILES']['MEMBERS']
members_messaged = config['FILES']['MEMBERS_MESSAGED']

import os

def save(df,path,name):
    if not os.path.exists(path):
        os.makedirs(path)
    n = r'\{}'.format(name)
    df.to_csv(path+n, index=False)

def load(path,name):
    n = r'\{}'.format(name)
    return pd.read_csv(path+n,encoding='iso-8859-1')

import datetime as dt
def GetWeekStart(week,year,weekday='-1'):
    d = str(year)+"-W"+str(week)
    r = dt.datetime.strptime(d + weekday, "%Y-W%W-%w")
    return r.strftime("%d-%b-%Y")

def readInputData(path):
    dataframes = []
    files = os.listdir(path)
    for filename in files:
        print(r"reading file: {}".format(filename))
        dataframes.append(load(path, filename))
    return pd.concat(dataframes)


path =  path+ r'{}'.format(version)
savepath = results_path+ r'{}'.format(version)

posts = load(path,'posts.csv')
comments = load(path,'comments.csv')


messages = posts

messages = messages.dropna(subset=['Message'])
msgs = messages['Message']
#detect post language
language = []
for post in msgs:
    language.append(detect_language(post,used_languages))

data = pd.DataFrame({'Text': msgs,'Language':language,'PostId':messages['Id'], 'GroupId':messages['GroupId'],'GroupName':messages['GroupName'], 'UserId':messages['UserId'], 'Created':messages['Created']})


save(data, savepath,r'PostsLanguages{}.csv'.format(version))

languages = data.groupby('Language')['PostId'].count().reset_index()

english = data[data['Language']=='english']
save(english,savepath,'EnglishPosts.csv')

import numpy as np
import plotly.graph_objs as go
import plotly.offline as ply
labels = languages.Language.values
values = languages.PostId.values
fig = {
  "data": [
    {
      "values": values,
      "labels": labels,
      "domain": {"x": [0, .48]},
      "name": "GHG Emissions",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    }],
  
}
ply.plot(fig, filename='donut')

activity = load(savepath,r'Activity{}.csv'.format(version))
posts = activity[activity.Type=='Comment'].reset_index()
messages = posts

messages = messages.dropna(subset=['Message'])
msgs = messages['Message']
#detect post language
language = []
for post in msgs:
    language.append(detect_language(post,used_languages))
    
data = pd.DataFrame({'Text': msgs,'Language':language,'PostId':messages['Id'], 'GroupId':messages['GroupId'],'GroupName':messages['GroupName'], 'UserId':messages['UserId']})
save(data, savepath,r'PostsLanguages{}.csv'.format(version))
english = data[data['Language']=='english']
save(english,savepath,'EnglishComments.csv')
