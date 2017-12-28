from learn_3 import *

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

Models.append('Normal Linear Regression')
reg = LinearRegression(n_jobs=-1)
reg.fit(train_[columns], train_['log_price'])
pred = np.exp(reg.predict(test_))
Accuracy = sqrt(mse(pred, test['price']))
print('\nRMSE for linear regression : ', Accuracy)



RMSE.append(Accuracy)
