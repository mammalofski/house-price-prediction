from learn_3 import *

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

pipe = Pipeline([
    ('sc', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=True)),
    ('las', Lasso())
])
model = GridSearchCV(pipe, param_grid={'las__alpha': [0.0005, 0.001, 0.01]})
model.fit(train_[columns], train_['log_price'])
degree = model.best_params_
print(degree)
pred = np.exp(model.predict(test_))
Accuracy = sqrt(mse(pred, test['price']))
print('==' * 20 + ' RMSE: ' + str(Accuracy) + ' ' + '==' * 20)
RMSE.append(Accuracy)
Models.append('Lasso')
