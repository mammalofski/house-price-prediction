from learn_3 import *

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('sc', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=True)),
    ('reg', LinearRegression())
])
model = GridSearchCV(pipe, param_grid={'poly__degree': [2, 3]})
model.fit(train_[columns], train_['log_price'])
degree = model.best_params_
print(degree)
pred = np.exp(model.predict(test_))
Accuracy = sqrt(mse(pred, test['price']))
print('==' * 20 + 'RMSE: ' + str(Accuracy) + '==' * 20)
RMSE.append(Accuracy)

Models.append('LinearRegression Step2 Polynominal')
