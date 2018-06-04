import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
iris=datasets.load_iris()
knn=KNeighborsClassifier(n_neighbors=7)
X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],test_size=0.3,random_state=42,stratify=iris['target'])
#print(X_test.shape)
#print(y_test.shape)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(knn.score(X_test,y_test))
plt.plot(pred)
