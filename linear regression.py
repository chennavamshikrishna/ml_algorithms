import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset=pd.read_csv('Salary_Data.csv')
#print(dataset)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=1/3)
##print(X_train)
#print(y_train)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,lin_reg.predict(X_train),c='blue')
plt.show()
print(lin_reg.score(X_train,y_train))


