
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
tweet = pd.read_csv("D:/BLR10AM/Assi/14.naive bayes/Datasets_Naive Bayes/Disaster_tweets_NB.csv",encoding = "ISO-8859-1")



#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["Id is just indexing ,irrelevant",
                "keyword types ,irrelevant",
                "place location ,irrelevant",
                 "textual data ,so important",
                 "given tweet is Fake or Real about real disaster occurring ,important",
                 ]

d_types =["count","nominal","nominal","textual","binary"]

data_details =pd.DataFrame({"column name":tweet.columns,
                            "data types ":d_types,
                            "description":description})

   #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc


import re
stop_words = []
# Load the custom built Stopwords
with open("D:/BLR10AM/codes/12.text mining/Datasets NLP/stopwords_en.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))



tweet.text = tweet.text.apply(cleaning_text)

# removing empty rows
tweet = tweet.loc[tweet.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

tweet_train, tweet_test = train_test_split(tweet, test_size = 0.25,random_state=7)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of tweet texts into word count matrix format - Bag of Words
tweets_bow = CountVectorizer(analyzer = split_into_words).fit(tweet.text)

# Defining BOW for all messages
all_tweets_matrix = tweets_bow.transform(tweet.text)

# For training messages
train_tweets_matrix = tweets_bow.transform(tweet_train.text)

# For testing messages
test_tweets_matrix = tweets_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire tweets
tfidf_transformer = TfidfTransformer().fit(all_tweets_matrix)

# Preparing TFIDF for train tweets
train_tfidf = tfidf_transformer.transform(train_tweets_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test tweets
test_tfidf = tfidf_transformer.transform(test_tweets_matrix)
test_tfidf.shape #  (row, column)



# Preparing a naive bayes model on training data set 
"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform the Naïve Bayes Machine Learning Algorithm
5.3	Validate the train and test data and perform confusion matrix, get precision, recall and accuracy from it. 
""" 



from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.



classifier_mb_lap = MB(alpha = 4)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap


#lets take alpaha =7
classifier_mb_lap = MB(alpha = 7)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap
