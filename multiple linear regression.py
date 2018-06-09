import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor2=LinearRegression()
regressor2.fit(X_train,y_train)

y_pred=regressor2.predict(X_test)
#print(y_pred)
# backward elimination is used to remove irrelevant independent varaibles that are not affecting the model
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())
