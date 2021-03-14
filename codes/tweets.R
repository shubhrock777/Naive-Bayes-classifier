# Import the salary dataset
library(readr)
tweet <- read.csv(file.choose())

tweet$target <- factor(tweet$target)

# examine the target variable more carefully
str(tweet$target)
table(tweet$target)

# proportion of fake or not fake tweets
prop.table(table(tweet$target))

# build a corpus using the text mining (tm) package
install.packages("tm")
library(tm)

str(tweet$text)

tweet_corpus <- Corpus(VectorSource(tweet$text))
tweet_corpus <- tm_map(tweet_corpus, function(x) iconv(enc2utf8(x), sub='byte'))

# clean up the corpus using tm_map()
corpus_clean <- tm_map(tweet_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# create a document-term sparse matrix
tweet_dtm <- DocumentTermMatrix(corpus_clean)
head(tweet_dtm[1:10, 1:30])

# To view DTM we need to convert it into matrix first
dtm_matrix <- as.matrix(tweet_dtm)
str(dtm_matrix)

View(dtm_matrix[1:10, 1:20])

colnames(tweet_dtm)[1:50]

# creating training and test datasets
tweet_train <- tweet[1:5329, ]
tweet_test  <- tweet[5330:7613, ]

tweet_corpus_train <- corpus_clean[1:5329]
tweet_corpus_test  <- corpus_clean[5330:7613]

tweet_dtm_train <- tweet_dtm[1:5329, ]
tweet_dtm_test  <- tweet_dtm[5330:7613, ]

# check that the proportion of fake tweet is similar
prop.table(table(tweet$target))

prop.table(table(tweet_train$target))
prop.table(table(tweet_test$target))

# indicator features for frequent words
# dictionary of words which are used more than 5 times
tweet_dict <- findFreqTerms(tweet_dtm_train, 5)

tweet_dtm_train <- DocumentTermMatrix(tweet_corpus_train, list(dictionary = tweet_dict))
tweet_dtm_test  <- DocumentTermMatrix(tweet_corpus_test, list(dictionary = tweet_dict))

tweet_test_matrix <- as.matrix(tweet_dtm_test)
View(tweet_test_matrix[1:10,1:10])

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
tweet_dtm_train <- apply(tweet_dtm_train, MARGIN = 2, convert_counts)
tweet_dtm_test  <- apply(tweet_dtm_test, MARGIN = 2, convert_counts)

View(tweet_dtm_train[1:10,1:10])

#Training a model on the data
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
tweet_classifier <- naiveBayes(tweet_dtm_train, tweet_train$target)
tweet_classifier

### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
tweet_lap <- naiveBayes(tweet_dtm_train, tweet_train$target,laplace = 2)
tweet_lap

##  Evaluating model performance with out laplace
tweet_test_pred <- predict(tweet_classifier, tweet_dtm_test)

# Evaluating model performance after applying laplace smoothing
tweet_test_pred_lap <- predict(tweet_lap, tweet_dtm_test)

## crosstable without laplace
library(gmodels)

CrossTable(tweet_test_pred, tweet_test$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(tweet_test_pred == tweet_test$target)
test_acc

## crosstable of laplace smoothing model
CrossTable(tweet_test_pred_lap, tweet_test$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(tweet_test_pred_lap == tweet_test$target)
test_acc_lap

# On Training Data without laplace 
tweet_train_pred <- predict(tweet_classifier, tweet_dtm_train)
tweet_train_pred

# train accuracy
train_acc = mean(tweet_train_pred == tweet_train$target)
train_acc

# prediction on train data for laplace model
tweet_train_pred_lap <- predict(tweet_lap, tweet_dtm_train)
tweet_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(tweet_train_pred_lap == tweet_train$target)
train_acc_lap