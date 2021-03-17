import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import sklearn
from pandas import Series
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



#####################################################################################################
#create a 10 fold cross-validation scheme for validation
folds = KFold(n_splits = 10, shuffle = True, random_state = 100)

# specify range of hyperparameters to tune (maximum number of features)
length_for_params=len(train_x.columns)+1
hyper_params = [{'n_features_to_select': list(range(1, length_for_params))}]

#specify model which must be trained
lr = LinearRegression()
lr.fit(train_x, train_y)
rfe = RFE(lr)

# train model based on hyperparameter tuning, calculate negative mean squared error per model and display in a results dataframe
model_cv = GridSearchCV(estimator = rfe,
                        param_grid = hyper_params,
                        scoring= 'neg_mean_squared_error',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

model_cv.fit(train_x, train_y)
cv_results = pd.DataFrame(model_cv.cv_results_)

#choose optimal number of features to maximise model performance
max_performance_feature=cv_results.loc[cv_results['mean_train_score'] == cv_results['mean_train_score'].max(), 'param_n_features_to_select'].iloc[0]
n_features_optimal = max_performance_feature

#train model based on the chosen number of features
lr = LinearRegression()
lr.fit(train_x, train_y)

rfe = RFE(lr, n_features_to_select=n_features_optimal)
rfe = rfe.fit(train_x, train_y)

#define evaluation and prediction for final test
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

# make final prediction and evaluate the performance by calling the evaluation function
base = evaluate(rfe, test_x, test_y)