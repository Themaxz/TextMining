
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
import os
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from subprocess import check_output

get_ipython().magic('matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# main_with_ini.py
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

secret_key = config['DEFAULT']['SECRET_KEY'] # 'secret-key-of-myapp'
ci_hook_url = config['CI']['HOOK_URL'] # 'web-hooking-url-from-ci-service'

# main_with_json.py
import json

with open('config.json', 'r') as f:
    config = json.load(f)

secret_key = config['DEFAULT']['SECRET_KEY'] # 'secret-key-of-myapp'
ci_hook_url = config['CI']['HOOK_URL'] # 'web-hooking-url-from-ci-service'
root_dir = config['FOLDERS']['ROOT_DIR']
path = config['FOLDERS']['PATH']
version = config['VERSION']
results_path = config['FOLDERS']['RESULTS']
version = config['VERSION']
members_list = config['FILES']['MEMBERS']
members_messaged = config['FILES']['MEMBERS_MESSAGED']


def save(df,path,name):
    n = r'\{}'.format(name)
    df.to_csv(path+n, index=False)

def load(path,name):
    n = r'\{}'.format(name)
    return pd.read_csv(path+n,encoding='iso-8859-1')
def GetWeekStart(week,year,weekday='-1'):
    d = str(year)+"-W"+str(week)
    r = dt.datetime.strptime(d + weekday, "%Y-W%W-%w")
    return r.strftime("%d-%b-%Y")




path =  path+ r'{}'.format(version)
savepath = results_path+ r'{}'.format(version)

posts = load(path,'posts.csv')
comments = load(path,'comments.csv')

if not os.path.exists(savepath):
    os.makedirs(savepath)



messages = posts

messages = messages.dropna(subset=['Message'])
rows = list(messages.index)
posts = posts.iloc[rows,:]

df = pd.read_csv(path+r'/training/movie_data.csv')
#randomizing set
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

#dividing training set from test set
train, test = train_test_split(df, test_size=0.2)

reviews = []
stopwords_set = set(stopwords.words("english"))

    
    

train_pos = train[ train['sentiment'] == 1]
train_pos = train_pos['review']
train_neg = train[ train['sentiment'] == 0]
train_neg = train_neg['review']


# In[43]:



for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.review.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    reviews.append((words_cleaned,row.sentiment))
    
    
# Extracting word features
def get_words_in_texts(text):
    all = []
    for (words, sentiment) in text:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

w_features = get_word_features(get_words_in_texts(reviews))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features


# In[44]:


w_features = get_word_features(get_words_in_texts(reviews))



# import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.review)



# In[45]:


#TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


# In[46]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[47]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train.sentiment)


# In[48]:


# Naive Baies
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf.fit(train.review, train.sentiment)  

#accuracy on test data
import numpy as np
docs_test = test.review
predicted = text_clf.predict(docs_test)
np.mean(predicted == test.sentiment)            

#adding group id to messages
messages['GroupId'] = messages['Id'].apply(lambda x: x.split('_')[0])


# Prediction on  DataSet

messages = messages.reset_index(drop=True)

predicted = text_clf.predict(messages.Message)

x = { messages.Message[index] : element for index,element in enumerate(predicted) }
summary = pd.DataFrame(index = range(len(x)),data= list(x.items()))
summary.columns = ['Message', 'Prediction']
summary['Sentiment NB'] = np.where(summary['Prediction']==1, 'POSITIVE', 'NEGATIVE')
summary['GroupId'] = messages['GroupId']
summary['UserId'] = messages['UserId']


## Logistic regression
from sklearn.linear_model import LogisticRegression


#Logistic regression
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression()),
])
text_clf.fit(train.review, train.sentiment)  

#accuracy on test data
import numpy as np
docs_test = test.review
predicted = text_clf.predict(docs_test)
np.mean(predicted == test.sentiment)   

#prediction on Dataset

messages = messages.reset_index(drop=True)

predicted1 = text_clf.predict(messages.Message)
x = { messages.Message[index] : element for index,element in enumerate(predicted1) }


summaryLR = pd.DataFrame()


summaryLR["Message"] = messages.Message
summaryLR['Prediction'] = predicted1
predicted_proba = text_clf.predict_proba(messages.Message)
prob = []
for p in predicted_proba:
    prob.append(max(p))

summaryLR['Prediction Probability'] = prob
summaryLR['Sentiment LR'] = np.where(summaryLR['Prediction']==1, 'POSITIVE', 'NEGATIVE')
summaryLR['GroupId'] = messages['GroupId']
summaryLR['UserId'] = messages['UserId']


pd.DataFrame(index = range(len(x)), data= list(x.items()))


#Adding neutral sentiment

prediction = predicted_proba.tolist()
def predict_with_neutral (probabilities, threshold):
    result = []
    for prob in probabilities:
        if max(prob)>threshold:
            result.append(prob.index(max(prob)))
        else:
            result.append(2)
    return result
        
neutral_predictions = predict_with_neutral(prediction,0.7)


# In[181]:


summary = pd.DataFrame()
summary["Message"] = messages.Message
summary['Prediction'] = neutral_predictions
predicted_proba = text_clf.predict_proba(messages.Message)
prob = []
for p in predicted_proba:
    prob.append(max(p))

summary['Prediction Probability'] = prob
summary['Sentiment'] = np.where(summary['Prediction']==1, 'POSITIVE', 'NEGATIVE')
summary['Sentiment'] = np.where(summary['Prediction']==2, 'NEUTRAL', summary['Sentiment'])
summary['GroupId'] = messages['GroupId']
summary['UserId'] = messages['UserId']

summary['PostId'] = messages['Id']
summaryLR['PostId'] = messages['Id']


sentiments = summary.groupby('Sentiment').count().reset_index()
sentiments = sentiments['PostId']
plt.pie(list(sentiments))


summary.to_csv('Sentiment.csv', index=False)
summaryLR.to_csv('SentimentLR.csv', index=False)


messages['Prediction'] = summary['Prediction']
messages['Sentiment'] = summary['Sentiment']





messages.to_csv('Messages.csv', index=False)

