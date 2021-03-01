import pandas as pd

weather_avg = pd.read_csv("weather_avg.csv")
weather_avg['id']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
weather_avg_pivoted = weather_avg.pivot(index=None, columns='meter_id')
weather_avg_transposed = weather_avg_pivoted.transpose()
weather_avg_transposed2=weather_avg_transposed
weather_avg_transposed2['avg temp']= weather_avg_transposed2.max(axis=1)
weather_avg_correct=weather_avg_transposed2['avg temp']
