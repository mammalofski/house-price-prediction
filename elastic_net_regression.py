from learn_3 import *

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet
pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(degree=2,include_bias=True)),
('en',ElasticNet())
])
model = GridSearchCV(pipe,param_grid={'en__alpha':[0.005,0.01,0.05,0.1],'en__l1_ratio':[0.1,0.4,0.8]})
model.fit(train_[columns],train_['log_price'])
degree = model.best_params_
print(degree)
pred = np.exp(model.predict(test_))
Accuracy = sqrt(mse(pred,test['price']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)
Models.append('ElasticNet Regression')
