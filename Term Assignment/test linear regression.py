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



# Read final dataset
final_df = pd.read_csv('Datasets/final_dataset.csv')

# Variables
# ignore_columns = ["datetime", "meter_num_id", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
#                 "T10", "T11", "T12", "T13", "T14", "T15", "T16",
#                  "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
#                  "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
#                  "T45", "T46", "T47", "T48"]

ignore_columns = ["datetime", "meter_num_id"]

# #make max usage column
# final_df['max']=final_df[["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
#                            "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
#                             "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
#                             "T45", "T46", "T47", "T48"]].max(axis=1)
#
# #turn into list and perform actions so max of tomorrow is in row of today and turn back into column and add to df
# max_aslist=final_df["max"].tolist()
# max_aslist.pop(0)
# def average(list):
#     return sum(list) / len(list)
# max_aslist.append(average(max_aslist))
#
# max = max_aslist
# final_df.loc[:,"max"] = max


#label_columns=['max']

#make average usage column as extra predictor variable
final_df['mean']=final_df[["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                           "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                            "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                            "T45", "T46", "T47", "T48"]].mean(axis=1)



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

#####################################################################################################
#create a 10 fold cross-validation scheme
folds = KFold(n_splits = 10, shuffle = True, random_state = 100)

# specify range of hyperparameters to tune
length_for_params=len(train_x.columns)+1
hyper_params = [{'n_features_to_select': list(range(1, length_for_params))}]

#perform grid search
#specify model
lr = LinearRegression()
lr.fit(train_x, train_y)
rfe = RFE(lr)

# call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe,
                        param_grid = hyper_params,
                        scoring= 'r2',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

# fit the model
model_cv.fit(train_x, train_y)

cv_results = pd.DataFrame(model_cv.cv_results_)
print(cv_results)

# plotting the results to see which number of parameters is optimal while not overfitting based on r_squared value
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='lower right')
plt.show()

# creating the model for final test based on optimal number of parameters

a=cv_results.loc[cv_results['mean_train_score'] == cv_results['mean_train_score'].max(), 'param_n_features_to_select'].iloc[0]
n_features_optimal = a

lr = LinearRegression()
lr.fit(train_x, train_y)

rfe = RFE(lr, n_features_to_select=n_features_optimal)
rfe = rfe.fit(train_x, train_y)

#define evaluation
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    result['predictions'] = list(map(lambda x: max(x), predictions))
    r2 = r2_score(test_labels, predictions)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mse = mean_squared_error(test_labels, predictions)
    print('Model Performance')
    print('mean squared error', mse)
    print("r squared of final test is ", r2)
    [print('Accuracy = {:0.2f}%.'.format(*x)) for x in np.c_[accuracy]][0]
    return mse

# predict max usages of test_x and evaluate
base = evaluate(rfe, test_x, test_y)
