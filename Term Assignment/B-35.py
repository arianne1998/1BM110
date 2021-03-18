import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import sklearn
from pandas import Series
from sklearn.metrics import roc_curve, auc, accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


#DATA MERGING

# start with average temperature
# read csv file
weather_avg = pd.read_csv("Datasets/weather_avg.csv")

# add id number
weather_avg['id']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]

# make a new column with the meter_id for each date
weather_avg_pivoted = weather_avg.pivot(index=None, columns='meter_id')

# switch the columns and rows
weather_avg_transposed = weather_avg_pivoted.transpose()

# copy the transposed df. each row contains one value which is the average temperature. This value is stored in the column 'avg temp'
weather_avg_transposed2=weather_avg_transposed
weather_avg_transposed2['avg temp']= weather_avg_transposed2.max(axis=1)

# Create a pandas series for the avg temperature for each date and each meter_id
weather_avg_correct=weather_avg_transposed2['avg temp']

# Repeat the steps from the average temperature for the minimum temperature
weather_min = pd.read_csv("Datasets/weather_min.csv")
weather_min['id']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
weather_min_pivoted = weather_min.pivot(index=None, columns='meter_id')
weather_min_transposed = weather_min_pivoted.transpose()
weather_min_transposed2=weather_min_transposed
weather_min_transposed2['min temp']= weather_min_transposed2.max(axis=1)
weather_min_correct=weather_min_transposed2['min temp']

# Repeat the steps from the average temperature for the maximum temperature
weather_max = pd.read_csv("Datasets/weather_max.csv")
weather_max['id']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
weather_max_pivoted = weather_max.pivot(index=None, columns='meter_id')
weather_max_transposed = weather_max_pivoted.transpose()
weather_max_transposed2=weather_max_transposed
weather_max_transposed2['max temp']= weather_max_transposed2.max(axis=1)
weather_max_correct=weather_max_transposed2['max temp']

# Create a dataframe for the pandas series for the average temperature
df_weather_avg = weather_avg_correct.to_frame()

# Create final dataframe with average, minimum and maximum temperature
weather=df_weather_avg
weather['min temp']=weather_min_correct
weather['max temp']=weather_max_correct
weather_correct = weather.reset_index()
weather_correct.rename(columns={'level_0':'date'}, inplace=True)

# Read csv files with meter info and meter consumptions
info = pd.read_csv('Datasets/info.csv')
meter_consumption = pd.read_csv('Datasets/meter_consumption.csv')

# Merge info df and meter_consumption df to one df called meter_complete
meter_complete = pd.merge(info, meter_consumption, on='meter_id', how='left')

# Merge meter_complete df and weather_correct df to one df called complete_dataset
complete_dataset = pd.merge(meter_complete, weather_correct, on=['meter_id', 'date'], how='left')

# Write final dataset to csv
complete_dataset.to_csv("Datasets/Complete Dataset.csv")

###############################################################################################################################################################################
#DATA VISUALISATION


dataset = pd.read_csv('Datasets/dataset_cleaned.csv')
dataset['daily_max'] = dataset[['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','T16','T17','T18','T19','T20','T21','T22','T23','T24','T25','T26','T27','T28','T29','T30','T31','T32','T33','T34','T35','T36','T37','T38','T39','T40','T41','T42','T43','T44','T45','T46','T47','T48']].max(axis=1)
dataset['avg_consumption'] = (dataset['T1']+dataset['T2']+dataset['T3']+dataset['T4']+dataset['T5']+dataset['T6']+dataset['T7']+dataset['T8']+dataset['T9']+dataset['T10']+dataset['T11']+dataset['T12']+dataset['T13']+dataset['T14']+dataset['T15']+dataset['T16']+dataset['T17']+dataset['T18']+dataset['T19']+dataset['T20']+dataset['T21']+dataset['T22']+dataset['T23']+dataset['T24']+dataset['T25']+dataset['T26']+dataset['T27']+dataset['T28']+dataset['T29']+dataset['T30']+dataset['T31']+dataset['T32']+dataset['T33']+dataset['T34']+dataset['T35']+dataset['T36']+dataset['T37']+dataset['T38']+dataset['T39']+dataset['T40']+dataset['T41']+dataset['T42']+dataset['T43']+dataset['T44']+dataset['T45']+dataset['T46']+dataset['T47']+dataset['T48'])
#dataset['normalized_avg_consumption'] = preprocessing.normalize(dataset[['avg_consumption']])

min_max_scaler = preprocessing.MinMaxScaler()
avg_consumption_scaled = min_max_scaler.fit_transform(dataset[['avg_consumption']].values.astype(float))
df_normalized = pd.DataFrame(avg_consumption_scaled)
dataset['normalized']=df_normalized[0]

plt.figure()
daily_max_dwelling_type = sns.boxplot(x = 'dwelling_type',y= 'daily_max',data = dataset)

xlabels = ['bungalow','semi detached house','flat','detached house','terraced house']
plt.figure()
dwelling_type_plt = sns.boxplot(x = 'dwelling_type', y = 'avg_consumption',data = dataset)
dwelling_type_plt = dwelling_type_plt.set_xticklabels(labels = xlabels, rotation=10)
plt.savefig('Figures/boxplot_dwellingtype.png')

plt.figure()
num_occupants_plt = sns.boxplot(x = 'num_occupants',y='avg_consumption',data=dataset)
plt.savefig('Figures/boxplot_numoccupants.png')

plt.figure()
heating_fuel_plt = sns.boxplot(x='heating_fuel',y='avg_consumption',data=dataset)
plt.savefig('Figures/boxplot_heatingfuel.png')

plt.figure()
heating_temperature_plt = sns.boxplot(x='heating_temperature',y='avg_consumption',data=dataset)
plt.savefig('Figures/boxplot_heatingtemperature.png')

plt.figure()
wall_insulation_plt = sns.boxplot(x='wall_insulation',y='avg_consumption',data=dataset)
plt.savefig('Figures/boxplot_wallinsulation.png')

plt.figure()
efficient_lighting_percentage_plt = sns.boxplot(x='efficient_lighting_percentage',y='avg_consumption',data=dataset)
plt.savefig('Figures/boxplot_efficientlightingpercentage.png')

plt.figure()
avg_temp_plt = plt.scatter(dataset['avg temp'],dataset.avg_consumption,s=1)
avg_temp_plt = plt.xlabel('Average temperature')
avg_temp_plt = plt.ylabel('Daily consumption')
plt.savefig('Figures/scatterplot_avgtemp.png')

plt.figure()
min_temp_plt = plt.scatter(dataset['min temp'],dataset.avg_consumption,s=1)
min_temp_plt = plt.xlabel('Minimum temperature')
min_temp_plt = plt.ylabel('Daily consumption')
plt.savefig('Figures/scatterplot_mintemp.png')

plt.figure()
max_temp_plt = plt.scatter(dataset['max temp'],dataset.avg_consumption,s=1)
max_temp_plt = plt.xlabel('Maximum temperature')
max_temp_plt = plt.ylabel('Daily consumption')
plt.savefig('Figures/scatterplot_maxtemp.png')

plt.figure()
avg_temp_daily_max = plt.scatter(dataset['avg temp'],dataset['daily_max'],s=1)
plt.savefig('Figures/temp_daily30max.png')

x = dataset.groupby(['datetime']).agg({'daily_max':'mean'})
x_correct = x.reset_index()
plt.figure()
daily_max_date = plt.scatter(x_correct['datetime'],x_correct['daily_max'])
plt.savefig('Figures/date_daily30max.png')

f = dataset.groupby(['datetime']).agg({'avg_consumption':'mean'})
f_correct = f.reset_index()
plt.figure()
daily_max_date = plt.scatter(f_correct['datetime'],f_correct['avg_consumption'])
plt.savefig('Figures/date_avgconsumption.png')

y = dataset.groupby(['datetime']).agg({'avg temp':'mean'})
y_correct = y.reset_index()
plt.figure()
daily_max_date = plt.scatter(y_correct['datetime'],y_correct['avg temp'])
plt.savefig('Figures/date_temp_scatter.png')

z = dataset.groupby(['datetime']).agg({'normalized':'mean','avg temp':'mean'})
z_correct = z.reset_index()
plt.figure()
daily_avg_avg_temp_date = plt.scatter(z_correct['avg temp'],z_correct['normalized'])
plt.savefig('Figures/normalizedconsump_temp_day_scatter.png')

h = dataset.groupby(['datetime']).agg({'avg_consumption':'mean','avg temp':'mean'})
h_correct = h.reset_index()
plt.figure()
daily_avg_avg_temp_date = plt.scatter(h_correct['avg temp'],h_correct['avg_consumption'])
plt.savefig('Figures/temp_day_avgconsump_scatter.png')
plt.show()

####################################################################################################################################################
#DATA CLEANING

# Read csv files
data = pd.read_csv('Datasets/Complete Dataset.csv')

# determine what the maximum consumption value is of each meter on each date and make a df with id, date, and max consumption
consumption = pd.read_csv("Datasets/meter_consumption.csv")
consumption2 = consumption.drop(['meter_id', 'date'], axis=1)
consumption3 = consumption2.max(axis=1)
consumption4 = consumption[['meter_id', 'date']].copy()
consumption4['max_consumption'] = consumption3


# Add the max consumption to the data and delete rows that hav nan for max consumption
data2 = pd.merge(data, consumption4, on=['meter_id', 'date'], how='left')
data_cleaned = data2[data2['max_consumption'].notna()]

# Make the meter IDs numerical from 1 to 51
meter_ids=data[['meter_id']].drop_duplicates('meter_id')
meter_ids['meter_num_id']=range(1,52)
data_cleaned2 = pd.merge(data_cleaned, meter_ids, on=['meter_id'], how='left')

# Drop columns that are not needed anymore (meter_id (original), unnamed:0 (index to column), and max_consumption (has been used for what))
data_cleaned3 = data_cleaned2.drop(['meter_id','Unnamed: 0','max_consumption'], axis=1)

# Parse Date to datetime
datetimes = pd.to_datetime(data_cleaned3["date"], format="%Y/%m/%d")
data_cleaned3['datetime']=datetimes
data_cleaned3 = data_cleaned3.drop(['date'], axis=1)

# Write dataset to csv before one hot encoding
dataset_cleaned = data_cleaned3
dataset_cleaned.to_csv("Datasets/dataset_cleaned.csv")

# One hot encoding for dwelling type (5 types)
# 5 types so 5 ids needed
ids_5 = [1,2,3,4,5]

# making sure all dwelling types are spelled correctly by removing duplicates
dwellings = data_cleaned3[['dwelling_type']].drop_duplicates('dwelling_type')

#add ids to dwellingtypes
dwellings['id']=ids_5

# binarize the dwelling types and add the dwelling type to the dataframe
dwelling_bin = pd.get_dummies(dwellings.dwelling_type, prefix='Dwelling')
dwelling_series = pd.Series(dwellings['dwelling_type'])
dwelling_bin['dwelling_type']=dwelling_series

#merging new columns to cleaned dataset and removing dwelling_type column (redundent)
data_cleaned4 = pd.merge(data_cleaned3, dwelling_bin, on='dwelling_type', how='left')
data_cleaned4 = data_cleaned4.drop(['dwelling_type'], axis=1)

# The same steps for one hot encoding that are used for dwelling_type can be used for heating_fuel, hot_water_fuel, wall_insulation, heating_temperature, and efficient lighting percentage.
# One hot encoding for heating fuel (4 types)
ids_4 = [1,2,3,4]
fuels = data_cleaned4[['heating_fuel']].drop_duplicates('heating_fuel')
fuels['id']=ids_4
fuels_bin = pd.get_dummies(fuels.heating_fuel, prefix='Fuel_Type')
fuel_series = pd.Series(fuels['heating_fuel'])
fuels_bin['heating_fuel']=fuel_series
data_cleaned5 = pd.merge(data_cleaned4, fuels_bin, on='heating_fuel', how='left')
data_cleaned5 = data_cleaned5.drop(['heating_fuel'], axis=1)

# One hot encoding for hot water fuel (3 types)
ids_3 = [1,2,3]
hw_fuels = data_cleaned5[['hot_water_fuel']].drop_duplicates('hot_water_fuel')
hw_fuels['id']=ids_3
hw_fuels_bin = pd.get_dummies(hw_fuels.hot_water_fuel, prefix='hw_fuel_Type')
hw_fuel_series = pd.Series(hw_fuels['hot_water_fuel'])
hw_fuels_bin['hot_water_fuel']=hw_fuel_series

data_cleaned6 = pd.merge(data_cleaned5, hw_fuels_bin, on='hot_water_fuel', how='left')
data_cleaned6 = data_cleaned6.drop(['hot_water_fuel'], axis=1)

# One hot encoding for wall insulation (5 types)
wall_insulations = data_cleaned6[['wall_insulation']].drop_duplicates('wall_insulation')
wall_insulations['id']=ids_5
wall_insulation_bin = pd.get_dummies(wall_insulations.wall_insulation, prefix='wall_insulation')
wall_insulation_series = pd.Series(wall_insulations['wall_insulation'])
wall_insulation_bin['wall_insulation']=wall_insulation_series
data_cleaned7 = pd.merge(data_cleaned6, wall_insulation_bin, on='wall_insulation', how='left')
data_cleaned7 = data_cleaned7.drop(['wall_insulation'], axis=1)

# One hot encoding for heating temperature (4 types)
heating_temperature = data_cleaned7[['heating_temperature']].drop_duplicates('heating_temperature')
heating_temperature['id']=ids_4
heating_temperature_bin = pd.get_dummies(heating_temperature.heating_temperature, prefix='heating_temperature')
heating_temperature_series = pd.Series(heating_temperature['heating_temperature'])
heating_temperature_bin['heating_temperature']=heating_temperature_series
data_cleaned8 = pd.merge(data_cleaned7, heating_temperature_bin, on='heating_temperature', how='left')
data_cleaned8 = data_cleaned8.drop(['heating_temperature'], axis=1)

# Sort date per 2 months
data_cleaned8['month_number'] = pd.DatetimeIndex(data_cleaned8['datetime']).month
data_cleaned8.loc[data_cleaned8['month_number']<=2,'month_group'] = 0
data_cleaned8.loc[(data_cleaned8['month_number']<=4)&(data_cleaned8['month_number']>2),'month_group'] = 1
data_cleaned8.loc[(data_cleaned8['month_number']<=6)&(data_cleaned8['month_number']>4),'month_group'] = 2
data_cleaned8.loc[(data_cleaned8['month_number']<=8)&(data_cleaned8['month_number']>6),'month_group'] = 3
data_cleaned8.loc[(data_cleaned8['month_number']<=10)&(data_cleaned8['month_number']>8),'month_group'] = 4
data_cleaned8.loc[(data_cleaned8['month_number']<=12)&(data_cleaned8['month_number']>10),'month_group'] = 5
data_cleaned9 = data_cleaned8.drop(['month_number'],axis = 1)

# One hot encoding for months
month_groups = data_cleaned9[['month_group']].drop_duplicates('month_group')
month_groups_bin = pd.get_dummies(month_groups.month_group, prefix ='month_group')
month_groups_series = pd.Series(month_groups['month_group'])
month_groups_bin['month_group'] = month_groups_series
data_cleaned9_1 = pd.merge(data_cleaned9,month_groups_bin,on='month_group',how='left')
data_cleaned9_1 = data_cleaned9_1.drop(['month_group'],axis=1)

# Binarize loft insulation
data_cleaned9_1.loc[data_cleaned9_1['loft_insulation']=='y','loft_insulation_y']=1
data_cleaned9_1.loc[data_cleaned9_1['loft_insulation']=='n','loft_insulation_y']=0
data_cleaned9_1 = data_cleaned9_1.drop(['loft_insulation'],axis=1)

# Binarize boiler age
data_cleaned9_1.loc[data_cleaned9_1['boiler_age']=='old','boiler_age_new'] = 0
data_cleaned9_1.loc[data_cleaned9_1['boiler_age']=='new','boiler_age_new'] = 1
data_cleaned9_1 = data_cleaned9_1.drop(['boiler_age'],axis=1)

# Transferring efficient lighting percentage to ordinal numbers (4 types)
efficient_lighting_percentage = data_cleaned9_1[['efficient_lighting_percentage']].drop_duplicates('efficient_lighting_percentage')
efficient_lighting_percentage['Ordinal_efficient_lighting_percentage']=ids_4
data_cleaned10 = pd.merge(data_cleaned9_1, efficient_lighting_percentage, on='efficient_lighting_percentage', how='left')
data_cleaned10 = data_cleaned10.drop(['efficient_lighting_percentage'], axis=1)

# Catogarize household appliances into large appliances (Dishwasher, Freezer, Fridge freezer, Refrigerator, Tumble Dryer, Washing machine) and small appliances (game console, lapotp, PC, Router, Set top box, Tablet, TV) and sum the number of appliances
data_cleaned10['large_appliances']=data_cleaned10['dishwasher']+data_cleaned10['freezer']+data_cleaned10['fridge_freezer']+data_cleaned10['refrigerator']+data_cleaned10['tumble_dryer']+data_cleaned10['washing_machine']
data_cleaned10['small_appliances']=data_cleaned10['game_console']+data_cleaned10['laptop']+data_cleaned10['pc']+data_cleaned10['router']+data_cleaned10['set_top_box']+data_cleaned10['tablet']+data_cleaned10['tv']

# Drop rows appliances
data_cleaned10 = data_cleaned10.drop(['dishwasher', 'freezer', 'fridge_freezer', 'refrigerator', 'tumble_dryer', 'washing_machine', 'game_console', 'laptop', 'pc', 'router', 'set_top_box', 'tablet', 'tv'], axis=1)

# final dataset and remove NaN values
final_dataset = data_cleaned10
final_dataset = final_dataset.dropna()

# Write final dataset to csv
final_dataset.to_csv("Datasets/final_dataset.csv")

###############################################################################################################################################################################
#CREATE LINEAR REGRESSION MODEL

# Read final dataset twice
final_df = pd.read_csv('Datasets/final_dataset.csv')
final_df2 = pd.read_csv('Datasets/final_dataset.csv')


# Variables, specify which variables are not needed for prediction(ignore) and which variables will be predicted(label)
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
max_performance_score = cv_results['mean_train_score'].max()
max_performance_feature=cv_results.iloc[cv_results['mean_train_score'].idxmax()]['param_n_features_to_select']
n_features_optimal = max_performance_feature

# determine lowest amount of features that doesn't deviate more than 5% of the best score to obtain good performance while minimising noise
for i, score in list(enumerate(cv_results["mean_train_score"])):
    max_diff_percentage = (score / max_performance_score * 100) - 100
    if max_diff_percentage < 5:
        n_features_optimal = cv_results.iloc[i]['param_n_features_to_select']
        break


# plotting the results to see which number of parameters is optimal while not overfitting based on r_squared value
plt.figure(figsize=(16,6))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='lower right')
plt.axvline(x=n_features_optimal)
plt.show()

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
    r_squared = r2_score(test_labels, predictions)
    adj_r_squared = 1 - (1-r_squared)*(len(test_labels)-1)/(len(test_labels)-test_features.shape[1]-1)
    print(predictions_df)
    print('Model Performance')
    print('mean squared error', mse)
    print('root mean squared error', rmse)
    print('adjusted r squared value', adj_r_squared)
    print('based on this number of features:', n_features_optimal)
    return predictions_df

# make final prediction and evaluate the performance by calling the evaluation function
base = evaluate(rfe, test_x, test_y)


#############################################################################################################################################################
#CREATE RANDOM FOREST MODEL

# Read final dataset twice
final_df_RF = pd.read_csv('Datasets/final_dataset.csv')
final_df2_RF = pd.read_csv('Datasets/final_dataset.csv')

# Variables, specify which variables are not needed for prediction(ignore) and which variables will be predicted(label)
ignore_columns = ["datetime", "meter_num_id", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                 "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                 "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                 "T45", "T46", "T47", "T48"]


all_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                 "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                 "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                 "T45", "T46", "T47", "T48"]

final_df['max value'] = final_df_RF[["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
                                     "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
                                     "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43", "T44",
                                     "T45", "T46", "T47", "T48"]].max(axis=1)

label_columns = ["max value"]


# Remove columns which should be ignored
final_df = final_df_RF.drop(columns=ignore_columns)

# Split x (features) and y (labels) in separate dataframes
final_x = final_df_RF.copy()
final_x = final_x.drop(columns=label_columns)
final_y = final_df_RF.copy()[label_columns]

# Split dataframes into test and train with a ratio of 30% - 70%
train_x, test_x, train_y, test_y = train_test_split(final_x, final_y, test_size=.3, random_state=42)

train_y=np.ravel(train_y)
test_y=np.ravel(test_y)


# create grid for hyperparameter tuning, values are estimated based on the data to make a first estimation
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

# search of parameters using 5 fold cross validation (5 is used here instead of 10 to reduce required computation time) and 400 iterations
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=400, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the search model
rf_random.fit(train_x, train_y)

#define evaluation and prediction
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    predictions=np.ravel(predictions)
    predictions=predictions.tolist()
    predictions_df=pd.DataFrame({'predictions':predictions})

    predictions_df.insert(0, "datetime", final_df2_RF['datetime'])
    predictions_df.insert(0, "meter_num_id", final_df2_RF['meter_num_id'])
    predictions_df.insert(3, "max value", final_df_RF['max value'])

    #calculate performance measures
    mse = mean_squared_error(test_labels, predictions)
    rmse = mean_squared_error(test_labels, predictions, squared=False)
    r_squared = r2_score(test_labels, predictions)
    adj_r_squared = 1 - (1-r_squared)*(len(test_labels)-1)/(len(test_labels)-test_features.shape[1]-1)

    print(predictions_df)
    print('Model Performance')
    print('mean squared error', mse)
    print('root mean squared error', rmse)
    print('adjusted r squared value', adj_r_squared)
    return predictions_df


# make preliminary prediction and evaluate the performance by calling the evaluation function
best_model = rf_random.best_estimator_
random_mse = evaluate(best_model, train_x, train_y)


#retrieve best parameters of search conducted above and create parameters similar to these for second round of hyperparameter tuning
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

# search of parameters using 5 fold cross validation (5 is used here instead of 10 to reduce required computation time) and 400 iterations
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=400, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

# Fit the search model
rf_random.fit(train_x, train_y)

# make final prediction and evaluate the performance by calling the evaluation function
best_model = rf_random.best_estimator_
random_mse = evaluate(best_model, test_x, test_y)

# give the parameters which are used in the final optimal model
print("optimal parameters which are used in the final model", rf_random.best_params_)

