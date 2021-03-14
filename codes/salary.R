# Import the salary dataset
library(readr)
salary_train <- read.csv(file.choose())
salary_test <- read.csv(file.choose())

#pre processing training dataset
x <- salary_train$Salary
salary_train$Salary <- ifelse(x == " <=50K",1,2)
View(x)
cols <- c("age" ,"workclass" , "education" ,"educationno" , "maritalstatus" , "occupation","relationship","race" ,"sex", "native" ,"Salary")
salary_train[cols] <- lapply(salary_train[cols] , factor)
str(salary_train)

#pre processing testing dataset
y <- salary_test$Salary
salary_test$Salary <- ifelse(y == " <=50K",1,2)
View(y)
cols <- c("age" ,"workclass" , "education" ,"educationno" , "maritalstatus" , "occupation","relationship","race" ,"sex", "native" ,"Salary")
salary_test[cols] <- lapply(salary_test[cols] , factor)
str(salary_test)

# examine the Salary variable more carefully
str(salary_train$Salary)
table(salary_train$Salary)

# proportion of salary 0(salary = <=50K ) and 1 (salary = >50K )
prop.table(table(salary_train$Salary))

#Training a model on the data
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
model <- naiveBayes(Salary~., data = salary_train)
model
### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
model2 <- naiveBayes(Salary~., data = salary_train,laplace = 3)
model2

##  Evaluating model performance with out laplace
salary_test_pred <- predict(model, salary_test)

# Evaluating model performance after applying laplace smoothing
salary_test_pred_lap <- predict(model2, salary_test)

## crosstable without laplace
install.packages("gmodels")
library(gmodels)

CrossTable(salary_test_pred, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(salary_test_pred == salary_test$Salary)
test_acc

## crosstable of laplace smoothing model
CrossTable(salary_test_pred_lap, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(salary_test_pred_lap == salary_test$Salary)
test_acc_lap

# On Training Data without laplace 
salary_train_pred <- predict(model, salary_train)

# train accuracy
train_acc = mean(salary_train_pred == salary_train$Salary)
train_acc


# prediction on train data for laplace model
salary_train_pred_lap <- predict(model2, salary_train)
salary_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(salary_train_pred_lap == salary_train$Salary)
train_acc_lap