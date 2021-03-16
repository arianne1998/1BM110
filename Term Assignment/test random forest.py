import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Read final dataset
final_df = pd.read_csv('Datasets/final_dataset.csv')

# Variables
ignore_columns = ["datetime", "meter_num_id", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                  "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                  "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                  "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                  "T45", "T46", "T47", "T48"]

# make max usage column
final_df['max'] = final_df[
    ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
     "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
     "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
     "T45", "T46", "T47", "T48"]].max(axis=1)

#turn into list and perform actions so max of tomorrow is in row of today and turn back into column and add to df
max_aslist=final_df["max"].tolist()
max_aslist.pop(0)
def average(list):
    return sum(list) / len(list)
max_aslist.append(average(max_aslist))

max = max_aslist
final_df.loc[:,"max"] = max
print(final_df)

label_columns=['max']

# make average usage column as an extra predictor
final_df['mean'] = final_df[
    ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
     "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
     "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
     "T45", "T46", "T47", "T48"]].mean(axis=1)

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
train_y=np.ravel(train_y)
test_y=np.ravel(test_y)

#####################################################################################################
# Iteration grid creation
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=50)]

# Number of features to consider at every split
max_features = ['sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 20]

# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 5, 10, 15, 20]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the grid to search for the optimal parameters
# Create  base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 5 fold cross validation (to reduce required computation time)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

# Fit the search model
rf_random.fit(train_x, train_y)

#evaluate grid search search
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mse = mean_squared_error(test_labels, predictions)
    print('Model Performance')
    print('mean squared error', mse)
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return mse

# calculate outcome of optimal model
best_model = rf_random.best_estimator_
random_mse = evaluate(best_model, train_x, train_y)

# give the parameters which are used in the optimal model
print(rf_random.best_params_)


n_estimators_start=round(rf_random.best_params_.get('n_estimators')*0.8,0)
n_estimators_stop=round(rf_random.best_params_.get('n_estimators')*1.2,0)

max_depth_start=round(rf_random.best_params_.get('max_depth')*0.8,0)
max_depth_stop=round(rf_random.best_params_.get('max_depth')*1.2,0)

min_samples_split_1=round(rf_random.best_params_.get('min_samples_split')*0.8,0)
min_samples_split_2=round(rf_random.best_params_.get('min_samples_split')*0.9,0)
min_samples_split_3=round(rf_random.best_params_.get('min_samples_split'),0)
min_samples_split_4=round(rf_random.best_params_.get('min_samples_split')*1.1,0)
min_samples_split_5=round(rf_random.best_params_.get('min_samples_split')*1.2,0)

min_samples_leaf_1=round(rf_random.best_params_.get('min_samples_leaf')*0.8,0)
min_samples_leaf_2=round(rf_random.best_params_.get('min_samples_leaf')*0.9,0)
min_samples_leaf_3=round(rf_random.best_params_.get('min_samples_leaf'),0)
min_samples_leaf_4=round(rf_random.best_params_.get('min_samples_leaf')*1.1,0)
min_samples_leaf_5=round(rf_random.best_params_.get('min_samples_leaf')*1.2,0)

bootstrap_choice=rf_random.best_params_.get('bootstrap')
max_features_choice=rf_random.best_params_.get("max_features")

#####################################################################################################
# refine the search by making a new grid with parameters around the best parameters found above
# Iteration grid creation
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=n_estimators_start, stop=n_estimators_stop, num=100)]

# Number of features to consider at every split
max_features = max_features_choice

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(max_depth_start, max_depth_stop, num=20)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [min_samples_split_1, min_samples_split_2, min_samples_split_3, min_samples_split_4, min_samples_split_5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [min_samples_leaf_1, min_samples_leaf_2, min_samples_leaf_3, min_samples_leaf_4, min_samples_leaf_5]

# Method of selecting samples for training each tree
bootstrap = [bootstrap_choice]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the grid to search for the optimal parameters
# Create  base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 5 fold cross validation (to reduce required computation time)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

# Fit the search model
rf_random.fit(train_x, train_y)

#evaluate grid search search
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mse = mean_squared_error(test_labels, predictions)
    print('Model Performance')
    print('mean squared error', mse)
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return mse

# calculate outcome of optimal model
best_model = rf_random.best_estimator_
random_mse = evaluate(best_model, test_x, test_y)

# give the parameters which are used in the optimal model
print(rf_random.best_params_)


