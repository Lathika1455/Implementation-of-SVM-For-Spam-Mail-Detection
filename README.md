# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
   
2.Read the data frame using pandas.

3.Get the information regarding the null values present in the dataframe.

4.Split the data into training and testing sets.

5.Convert the text data into a numerical representation using CountVectorizer.

6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.

7.Finally, evaluate the accuracy of the model. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: LATHIKA SREE R
RegisterNumber: 212224040169
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect (rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,
random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

## Result output
<img width="791" height="39" alt="image" src="https://github.com/user-attachments/assets/c3fb7656-6227-42cb-8339-aed2af41731f" />

## data.head()
<img width="1027" height="312" alt="image" src="https://github.com/user-attachments/assets/90fbcc5e-d745-4636-bd4c-ffc3d851007b" />

## data.info()
<img width="697" height="269" alt="image" src="https://github.com/user-attachments/assets/7b11d152-718b-4771-95c3-04781cc74054" />

## data.isnull().sum()
<img width="608" height="281" alt="image" src="https://github.com/user-attachments/assets/c62be104-f8b3-4145-8a4a-8f60d1f208e8" />

## y_pred
<img width="799" height="48" alt="image" src="https://github.com/user-attachments/assets/f161a825-004e-4d8c-9bc2-9eeeea7e9971" />

## accuracy()
<img width="407" height="100" alt="image" src="https://github.com/user-attachments/assets/db63c67b-8e6b-42a8-ae38-7d5a6de1a3f3" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
