# Import the car_ad dataset
library(readr)
car_ad <- read.csv(file.choose())
car_ad <- car_ad[c(-1)]

#pre processing dataset
cols <- c("Gender" , "Age" ,"Purchased")
cols <- c("Purchased")
car_ad[cols] <- lapply(car_ad[cols] , factor)
car_ad[c(3)] <- scale(car_ad[c(3)])
view(car_ad)

# examine the Purchased variable more carefully
str(car_ad$Purchased)
table(car_ad$Purchased)

# proportion of Purchased
prop.table(table(car_ad$Purchased))

#data partition
set.seed(1234)
ind <- sample(2 , nrow(car_ad) , replace = TRUE , prob = c(0.8 , 0.2))
train <- car_ad[ind == 1 , ]
test <- car_ad[ind == 2 , ]

#Training a model on the data
#install.packages("e1071")
library(e1071)

## building naiveBayes classifier without laplace
model <- naiveBayes(Purchased~., data = train)
model

##  Evaluating model performance without laplace
test_pred <- predict(model, test)

## crosstable
#install.packages("gmodels")
library(gmodels)

CrossTable(test_pred, test$Purchased,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(test_pred == test$Purchased)
test_acc

# On Training Data 
train_pred <- predict(model, train)

# train accuracy
train_acc = mean(train_pred == train$Purchased)
train_acc

## building naiveBayes classifier with laplace
model <- naiveBayes(Purchased~., data = train , laplace =3)
model

##  Evaluating model performance with laplace
test_pred <- predict(model, test)


## test accuracy for laplace model
test_acc <- mean(test_pred == test$Purchased)
test_acc

# On Training Data 
train_pred <- predict(model, train)

# train accuracy
train_acc = mean(train_pred == train$Purchased)
train_acc