import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Datasets/dataset_cleaned.csv')

dataset['avg_consumption'] = (dataset['T1']+dataset['T2']+dataset['T3']+dataset['T4']+dataset['T5']+dataset['T6']+dataset['T7']+dataset['T8']+dataset['T9']+dataset['T10']+dataset['T11']+dataset['T12']+dataset['T13']+dataset['T14']+dataset['T15']+dataset['T16']+dataset['T17']+dataset['T18']+dataset['T19']+dataset['T20']+dataset['T21']+dataset['T22']+dataset['T23']+dataset['T24']+dataset['T25']+dataset['T26']+dataset['T27']+dataset['T28']+dataset['T29']+dataset['T30']+dataset['T31']+dataset['T32']+dataset['T33']+dataset['T34']+dataset['T35']+dataset['T36']+dataset['T37']+dataset['T38']+dataset['T39']+dataset['T40']+dataset['T41']+dataset['T42']+dataset['T43']+dataset['T44']+dataset['T45']+dataset['T46']+dataset['T47']+dataset['T48'])

x = dataset.groupby(['meter_num_id','dwelling_type']).aggregate({'avg_consumption':'mean'})

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
plt.savefig('Figures/scatterplot_maxtemp.png')
