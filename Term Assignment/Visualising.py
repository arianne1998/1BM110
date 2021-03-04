import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

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
dwelling_type_plt = dwelling_type_plt.set_xticklabels(labels = xlabels, rotation=30)
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
#plt.savefig('Figures/scatterplot_maxtemp.png')

plt.figure()
avg_temp_daily_max = plt.scatter(dataset['avg temp'],dataset['daily_max'],s=1)

x = dataset.groupby(['datetime']).agg({'daily_max':'mean'})
x_correct = x.reset_index()
plt.figure()
daily_max_date = plt.scatter(x_correct['datetime'],x_correct['daily_max'])

y = dataset.groupby(['datetime']).agg({'avg temp':'mean'})
y_correct = y.reset_index()
plt.figure()
daily_max_date = plt.scatter(y_correct['datetime'],y_correct['avg temp'])

z = dataset.groupby(['datetime']).agg({'normalized':'mean','avg temp':'mean'})
z_correct = z.reset_index()
plt.figure()
daily_avg_avg_temp_date = plt.scatter(z_correct['avg temp'],z_correct['normalized'])

h = dataset.groupby(['datetime']).agg({'avg_consumption':'mean','avg temp':'mean'})
h_correct = h.reset_index()
plt.figure()
daily_avg_avg_temp_date = plt.scatter(h_correct['avg temp'],h_correct['avg_consumption'])
plt.show()