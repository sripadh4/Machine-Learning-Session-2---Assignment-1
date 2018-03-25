# Machine-Learning-Session-2---Assignment-1

Build the linear regression model using scikit learn in boston data to predict 'Price' based on other dependent variable.

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split
lm = linear_model.LinearRegression()
boston = load_boston()
bos = pd.DataFrame(boston.data, columns=boston.feature_names)
Price_Prediction = pd.DataFrame(boston.target, columns=["PP"])
y = Price_Prediction["PP"]
bos_train, bos_test, y_train, y_test = train_test_split(bos, y, test_size=0.2, random_state=1)
Model = lm.fit(bos_train, y_train)
Prediction = lm.predict(bos_test)
print ('Coefficients: ', lm.coef_)
print ('Intercept: ', lm.intercept_)
print ('Coefficient of Determination: ', lm.score(bos_test, y_test))
plt.scatter(y_test, Prediction)
plt.show()
