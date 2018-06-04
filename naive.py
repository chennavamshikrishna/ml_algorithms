from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.3)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
sco=model.score(X_test,y_test)
print(y_pred)
print(sco)
y_pred.shape
import matplotlib.pyplot as plt
import  numpy as np
#plt.scatter(y_pred,y_train)
#plt.show()
