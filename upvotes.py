from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.io import arff
import pandas as pd
import gc
import statistics
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

temp=pd.DataFrame()

data = pd.read_csv('datasets/upvotes/train.csv')
test_data = pd.read_csv('datasets/upvotes/test.csv')


label = [0,1,2,3,4,5,6]
bins = [0,11,22,33,44,55,66,77]
data['Binned'] = pd.cut(data['Answers'], bins=bins, labels=label)
test_data['Binned'] = pd.cut(test_data['Answers'], bins=bins, labels=label)

labelencoder_X = LabelEncoder()
data['Tag'] = labelencoder_X.fit_transform(data['Tag'])
test_data['Tag'] = labelencoder_X.fit_transform(test_data['Tag'])

data = data.fillna(value=0)
test_data = test_data.fillna(value=0)
yVar = data.Upvotes

data = data.drop(['Upvotes'], axis=1)

temp['ID'] = test_data.ID
data = data.drop(['ID','Username'], axis=1)
test_data = test_data.drop(['ID','Username'], axis=1)

X_train=data
Y_train = yVar


clf1 = RandomForestRegressor(n_estimators='warn', criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)

clf1.fit(X_train, Y_train)

preds = clf1.predict(test_data)
excel=pd.DataFrame()
excel['ID'] = temp.ID
excel['Upvotes'] = preds.astype('int')
excel.to_csv("Upvotes(clean)5.csv", index=False)


# My Rank       179     Score   1079.6150956141
