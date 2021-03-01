import pandas as pd

# start with average temperatur
# read csv file
weather_avg = pd.read_csv("weather_avg.csv")

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
weather_min = pd.read_csv("weather_min.csv")
weather_min['id']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
weather_min_pivoted = weather_min.pivot(index=None, columns='meter_id')
weather_min_transposed = weather_min_pivoted.transpose()
weather_min_transposed2=weather_min_transposed
weather_min_transposed2['min temp']= weather_min_transposed2.max(axis=1)
weather_min_correct=weather_min_transposed2['min temp']

# Repeat the steps from the average temperature for the maximum temperature
weather_max = pd.read_csv("weather_max.csv")
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
info = pd.read_csv('info.csv')
meter_consumption = pd.read_csv('meter_consumption.csv')

# Merge info df and meter_consumption df to one df called meter_complete
meter_complete = pd.merge(info, meter_consumption, on='meter_id', how='left')

# Merge meter_complete df and weather_correct df to one df called complete_dataset
complete_dataset = pd.merge(meter_complete, weather_correct, on=['meter_id', 'date'], how='left')
