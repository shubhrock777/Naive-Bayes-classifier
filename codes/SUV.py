
import pandas as pd 

#loading the dataset

df = pd.read_csv("D:/BLR10AM/Assi/14.naive bayes/Datasets_Naive Bayes/NB_Car_Ad.csv")



#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["userid is unique customer number ,irrelevant",
                "Gender is the range of characteristics pertaining to, and differentiating between, femininity and masculinity,important",
                "age of person , impotant",
                "salary of person,very important",
                "did he purchased the SUV or not"]

d_types =["nominal","binary","ratio","ratio","binary"]

data_details =pd.DataFrame({"column name":df.columns,
                            "data types ":d_types,
                            "description":description})

            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of df 
df.info()
df.describe()          


#data types        
df.dtypes


#checking for na value
df.isna().sum()
df.isnull().sum()

#checking unique value for each columns
df.nunique()

#variance of df
df.var()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}

EDA

# covariance for data set 
covariance = df.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(df.iloc[:, :])


#boxplot for every columns
df.columns
df.nunique()

boxplot = df.boxplot(column=["Age","EstimatedSalary"])   #no outlier

#creatind dataframe with only with (discrete,continuous ,output)
X_c = df.iloc[:,[2,3]]
X_d = df.iloc[:,[1]]
Y   = df.iloc[:,[4]]

# Normalization functio,
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# Normalized data frame (considering the numerical part of data)
X_c = norm_func(X_c.iloc[:,:])


#dummies for discrete data
df_dummies = pd.get_dummies(X_d)

#  creating a new df with both df 
X = pd.concat([df_dummies, X_c], axis=1)




from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=3)

#converting ouput into  series
y_test = Y_test["Purchased"]

y_train = Y_train["Purchased"]


""" 5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform the Naïve Bayes Machine Learning Algorithm
5.3	Validate the train and test data and perform confusion matrix, get precision, recall and accuracy from it. 
 

"""
# Preparing a naive bayes model on training data set 
import numpy as np
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(X_train, Y_train)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(X_test)
accuracy_test_m = np.mean(test_pred_m == y_test)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, Y_test) 

pd.crosstab(test_pred_m, y_test)

# Training Data accuracy
train_pred_m = classifier_mb.predict(X_train)
accuracy_train_m = np.mean(train_pred_m == y_train)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.



classifier_mb_lap = MB(alpha =4 )
classifier_mb_lap.fit(X_train, Y_train )

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(X_test)
accuracy_test_lap = np.mean(test_pred_lap == y_test)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, Y_test) 

pd.crosstab(test_pred_lap, y_test)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(X_train)
accuracy_train_lap = np.mean(train_pred_lap == y_train)
accuracy_train_lap
