from pprint import pprint

from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

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

# Create and train LogisticRegression model
lrm = LinearRegression()

lrm.fit(train_x, train_y)

# Predicting the Test set results
y_pred = lrm.predict(test_x)
result = test_x.copy()
result['predictions'] = list(map(lambda x: max(x), y_pred))

print("LinearRegression Result:")
print(result.head())
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
print("\n\n")

# RandomForest
sc = StandardScaler()
scaled_train_x = sc.fit_transform(train_x)
scaled_test_x = sc.transform(test_x)

# TODO determine best Tree depth
rfr = RandomForestRegressor(n_estimators=20, random_state=0)
rfr.fit(scaled_train_x, train_y)
y_pred = rfr.predict(scaled_test_x)
# Not sure if we can use test_x here while predicting with on scaled_test_x
result = test_x.copy()
result['predictions'] = list(map(lambda x: max(x), y_pred))

print("RandomForest Result:")
print(result.head())
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
print("\n\n")
