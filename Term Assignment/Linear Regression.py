from pprint import pprint

from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# import csv files into dataframes
info_df = pd.read_csv('Datasets/info.csv')
meter_consumption_df = pd.read_csv('Datasets/meter_consumption.csv')
weather_avg_df = pd.read_csv('Datasets/weather_avg.csv')
weather_max_df = pd.read_csv('Datasets/weather_max.csv')
weather_min_df = pd.read_csv('Datasets/weather_min.csv')

# Read final dataset
final_df = pd.read_csv('Datasets/Final Dataset.csv')

# Variables
ignore_columns = ["boiler_age", "loft_insulation", "datetime"]
label_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                 "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                 "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                 "T45", "T46", "T47", "T48"]

# Remove columns which should be ignored
final_df = final_df.drop(columns=ignore_columns)
# TODO fix this in the cleaning process
final_df = final_df.dropna()

# Split x (features) and y (labels) in separate dataframes
final_x = final_df.copy()
final_x = final_x.drop(columns=label_columns)
final_y = final_df.copy()[label_columns]


# Split dataframes into test and train with a ratio of 30%
train_x, test_x, train_y, test_y = train_test_split(final_x, final_y, test_size=.3, random_state=0)

#Training data is tuned with Hyperparameter Tuning Using Grid Search Cross-Validation

#Step 1: creating a 10-kfold
folds = KFold(n_splits = 10, shuffle = True, random_state = 100)

#Step 2: specifying the number of hyperparameters to tune, based on number of predictors.
number_of_features = len(train_x.columns) + 1
hyper_params = [{'n_features_to_select': list(range(1,number_of_features))}]

#Step 3: performing the Grid Search Cross-Validation
#creating linear regression model
lmr = LinearRegression()
lmr.fit(train_x, train_y)

#
rfe = RFE(lmr)

#implement the GridSearchCV()
model_cv = GridSearchCV(estimator = rfe,
                        param_grid = hyper_params,
                        scoring= 'neg_root_mean_squared_error',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

# fit the model
model_cv.fit(train_x, train_y)

cv_results = pd.DataFrame(model_cv.cv_results_)
print(cv_results)

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

# tuples of (feature name, whether selected, ranking)
# note that the 'rank' is > 1 for non-selected features
#list(zip(train_x.columns,rfe.support_,rfe.ranking_))

# final model
n_features_optimal = 10

lmr = LinearRegression()
lmr.fit(train_x, train_y)

rfe = RFE(lmr, n_features_to_select=n_features_optimal)
rfe = rfe.fit(train_x, train_y)

# predict prices of X_test
y_pred = lmr.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

print("LinearRegression Result:")
print(result.head())
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
print("\n\n")










