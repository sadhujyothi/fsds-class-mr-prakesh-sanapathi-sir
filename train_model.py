#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset..
dataset=pd.read_csv(r"C:\Users\HP\Downloads\student_info.csv")

dataset

# head of 5 rows..

dataset.head()

#last of 5 rows

dataset.tail()

# shape of dataset

dataset.shape

# discover and visualize the data to gain insights

dataset.info()


# descriptive statistics
dataset.describe()

# create a ploat

plt.scatter(x=dataset.study_hours, y=dataset.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Student marks")
plt.title("Scatter Plot of the Student Study Hours vs Students marks")
plt.show()

# prepare the data for machine learning algorithms

#data cleaning

dataset


dataset.isnull().sum()

dataset.mean()

dataset2=dataset.fillna(dataset.mean())
dataset2.isnull().sum()

dataset2.head()
dataset.head()

dataset.info()
dataset2.info()

#split dataset

x=dataset2.drop("student_marks",axis="columns")
y=dataset2.drop("study_hours",axis="columns")
print("shape of x=",x.shape)
print("shape of y=",y.shape)

x

y


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)

print("shape of X_train = ", x_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of X_test = ", x_test.shape)
print("shape of y_test = ", y_test.shape)

x_train

x_test.shape

y_train

y_test



#select a model and train it

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr

lr.fit(x_train,y_train)

lr.get_params()

lr.coef_

lr.intercept_


m=3.93
c=50.44
y=m*10+c
y

m=3.93
c=50.44
y=m*11+c
y

lr.predict([[11]])
lr


y_pred = lr.predict(x_test)
y_pred

pd.DataFrame(np.c_[x_test,y_test    ,y_pred],columns=["study_hours","student_marks_original","student_marks_predicted"])

lr

lr.score(x_test,y_test) 

lr.score(x_train,y_train)

plt.scatter(x_train,y_train) 


plt.scatter(x_test, y_test)
plt.plot(x_train, lr.predict(x_train), color = "r")

lr


#create pipeline

import joblib # create pipeline 
joblib.dump(lr, "Desktop.pkl")

model = joblib.load("Desktop.pkl") 

__name__
