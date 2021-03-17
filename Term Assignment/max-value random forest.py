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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


# Read final dataset twice
final_df = pd.read_csv('Datasets/final_dataset.csv')
final_df2 = pd.read_csv('Datasets/final_dataset.csv')

# Variables, specify which variables are not needed for prediction and which variables will be predicted
ignore_columns = ["datetime", "meter_num_id", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                 "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                 "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                 "T45", "T46", "T47", "T48"]


all_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                 "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                 "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                 "T45", "T46", "T47", "T48"]

final_df['max value'] = final_df[["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                                     "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                                     "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                                     "T45", "T46", "T47", "T48"]].max(axis=1)

label_columns = ["max value"]


# Remove columns which should be ignored
final_df = final_df.drop(columns=ignore_columns)

# Split x (features) and y (labels) in separate dataframes
final_x = final_df.copy()
final_x = final_x.drop(columns=label_columns)
final_y = final_df.copy()[label_columns]

# Split dataframes into test and train with a ratio of 30% - 70%
train_x, test_x, train_y, test_y = train_test_split(final_x, final_y, test_size=.3, random_state=42)

train_y=np.ravel(train_y)
test_y=np.ravel(test_y)

#####################################################################################################
# create grid for hyperparameter tuning, values are somewhat randomly sampled to make a first estimation
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=25)]

# Number of features to consider at every split
max_features = ['sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [10, 15, 20, 25, 30]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10, 15]

# Method of selecting samples for training each tree, with or without replacement
bootstrap = [True, False]

# Create the random grid so it can be called upon later
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the grid to search for the optimal parameters given the input
# Create  base model to tune
rf = RandomForestRegressor()

# search of parameters using 5 fold cross validation (5 is used here instead of 10 to reduce required computation time)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=400, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the search model
rf_random.fit(train_x, train_y)

#define evaluation and prediction for preliminary test
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    predictions=np.ravel(predictions)
    predictions=predictions.tolist()
    predictions_df=pd.DataFrame({'predictions':predictions})

    predictions_df.insert(0, "datetime", final_df2['datetime'])
    predictions_df.insert(0, "meter_num_id", final_df2['meter_num_id'])
    predictions_df.insert(3, "max value", final_df['max value'])

    #calculate performance measures
    mse = mean_squared_error(test_labels, predictions)
    rmse = mean_squared_error(test_labels, predictions, squared=False)

    print(predictions_df)
    print('Model Performance')
    print('mean squared error', mse)
    print('root mean squared error', rmse)
    return predictions_df


# make preliminary prediction and evaluate the performance by calling the evaluation function
best_model = rf_random.best_estimator_
random_mse = evaluate(best_model, train_x, train_y)

# give the parameters which are used in the preliminary model
print(rf_random.best_params_)

#retrieve best parameters of search conducted above and create parameters similar to these for new hyperparameter tuning
n_estimators_start=int(round(rf_random.best_params_.get('n_estimators')*0.8,0))
n_estimators_stop=int(round(rf_random.best_params_.get('n_estimators')*1.2,0))

max_depth_start=int(round(rf_random.best_params_.get('max_depth')*0.8,0))
max_depth_stop=int(round(rf_random.best_params_.get('max_depth')*1.2,0))

min_samples_split_1=int(round(rf_random.best_params_.get('min_samples_split')*0.8,0))
min_samples_split_2=int(round(rf_random.best_params_.get('min_samples_split')*0.9,0))
min_samples_split_3=int(round(rf_random.best_params_.get('min_samples_split'),0))
min_samples_split_4=int(round(rf_random.best_params_.get('min_samples_split')*1.1,0))
min_samples_split_5=int(round(rf_random.best_params_.get('min_samples_split')*1.2,0))

min_samples_leaf_1=int(round(rf_random.best_params_.get('min_samples_leaf')*0.8,0))
min_samples_leaf_2=int(round(rf_random.best_params_.get('min_samples_leaf')*0.9,0))
min_samples_leaf_3=int(round(rf_random.best_params_.get('min_samples_leaf'),0))
min_samples_leaf_4=int(round(rf_random.best_params_.get('min_samples_leaf')*1.1,0))
min_samples_leaf_5=int(round(rf_random.best_params_.get('min_samples_leaf')*1.2,0))

bootstrap_choice=rf_random.best_params_.get('bootstrap')


#####################################################################################################
# refine the search by making a new grid with parameters around the best parameters found above
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=n_estimators_start, stop=n_estimators_stop, num=25)]

# Number of features to consider at every split
max_features = ['sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(max_depth_start, max_depth_stop, num=10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [min_samples_split_1, min_samples_split_2, min_samples_split_3, min_samples_split_4, min_samples_split_5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [min_samples_leaf_1, min_samples_leaf_2, min_samples_leaf_3, min_samples_leaf_4, min_samples_leaf_5]

# Method of selecting samples for training each tree
bootstrap = [bootstrap_choice]

# Create the random grid so it can be called upon later
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the grid to search for the optimal parameters
# Create  base model to tune
rf = RandomForestRegressor()

# search of parameters using 5 fold cross validation (5 is used here instead of 10 to reduce required computation time)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=400, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

# Fit the search model
rf_random.fit(train_x, train_y)

# make final prediction and evaluate the performance by calling the evaluation function
best_model = rf_random.best_estimator_
random_mse = evaluate(best_model, test_x, test_y)

# give the parameters which are used in the final optimal model
print(rf_random.best_params_)


