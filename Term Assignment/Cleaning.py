import pandas as pd

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

# Transferring efficient lighting percentage to ordinal numbers (4 types)
efficient_lighting_percentage = data_cleaned8[['efficient_lighting_percentage']].drop_duplicates('efficient_lighting_percentage')
efficient_lighting_percentage['Ordinal_efficient_lighting_percentage']=ids_4
data_cleaned9 = pd.merge(data_cleaned8, efficient_lighting_percentage, on='efficient_lighting_percentage', how='left')
data_cleaned9 = data_cleaned9.drop(['efficient_lighting_percentage'], axis=1)

# Catogarize household appliances into large appliances (Dishwasher, Freezer, Fridge freezer, Refrigerator, Tumble Dryer, Washing machine) and small appliances (game console, lapotp, PC, Router, Set top box, Tablet, TV) and sum the number of appliances
data_cleaned9['large_appliances']=data_cleaned9['dishwasher']+data_cleaned9['freezer']+data_cleaned9['fridge_freezer']+data_cleaned9['refrigerator']+data_cleaned9['tumble_dryer']+data_cleaned9['washing_machine']
data_cleaned9['small_appliances']=data_cleaned9['game_console']+data_cleaned9['laptop']+data_cleaned9['pc']+data_cleaned9['router']+data_cleaned9['set_top_box']+data_cleaned9['tablet']+data_cleaned9['tv']

# Drop rows appliances
data_cleaned9 = data_cleaned9.drop(['dishwasher', 'freezer', 'fridge_freezer', 'refrigerator', 'tumble_dryer', 'washing_machine', 'game_console', 'laptop', 'pc', 'router', 'set_top_box', 'tablet', 'tv'], axis=1)

# final dataset
final_dataset = data_cleaned9


# Write final dataset to csv
final_dataset.to_csv("Datasets/Final Dataset.csv")
