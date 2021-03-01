import pandas as pd
import datetime as dt

# Read csv files
data = pd.read_csv('Complete Dataset.csv')

# determine what the maximum consumption value is of each meter on each date and make a df with id, date, and max consumption
consumption = pd.read_csv("meter_consumption.csv")
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

# One hot encoding for dwelling type (5 types)
ids = [1,2,3,4,5]
dwellings = data_cleaned3[['dwelling_type']].drop_duplicates('dwelling_type')
dwellings['id']=ids
y = pd.get_dummies(dwellings.dwelling_type, prefix='Dwelling')