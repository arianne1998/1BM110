import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv('Datasets/dataset_cleaned.csv')

dataset['avg_consumption'] = (dataset['T1']+dataset['T2']+dataset['T3']+dataset['T4']+dataset['T5']+dataset['T6']+dataset['T7']+dataset['T8']+dataset['T9']+dataset['T10']+dataset['T11']+dataset['T12']+dataset['T13']+dataset['T14']+dataset['T15']+dataset['T16']+dataset['T17']+dataset['T18']+dataset['T19']+dataset['T20']+dataset['T21']+dataset['T22']+dataset['T23']+dataset['T24']+dataset['T25']+dataset['T26']+dataset['T27']+dataset['T28']+dataset['T29']+dataset['T30']+dataset['T31']+dataset['T32']+dataset['T33']+dataset['T34']+dataset['T35']+dataset['T36']+dataset['T37']+dataset['T38']+dataset['T39']+dataset['T40']+dataset['T41']+dataset['T42']+dataset['T43']+dataset['T44']+dataset['T45']+dataset['T46']+dataset['T47']+dataset['T48'])

x = dataset.groupby(['meter_num_id','dwelling_type']).aggregate({'avg_consumption':'mean'})

x1 = dataset.avg_consumption[dataset['dwelling_type']=='bungalow']
x2 = dataset.avg_consumption[dataset['dwelling_type']=='semi_detached_house']
x3 = dataset.avg_consumption[dataset['dwelling_type']=='detached_house']
x4 = dataset.avg_consumption[dataset['dwelling_type']=='flat']
x5 = dataset.avg_consumption[dataset['dwelling_type']=='terraced_house']
xlabels = ['bungalow','semi detached house','detached house','flat','terraced house']
#_ = plt.boxplot([x1,x2,x3,x4,x5],labels = labels)

y1 = dataset.avg_consumption[dataset['num_occupants']==1.0]
y2 = dataset.avg_consumption[dataset['num_occupants']==2.0]
y3 = dataset.avg_consumption[dataset['num_occupants']==3.0]
y4 = dataset.avg_consumption[dataset['num_occupants']==4.0]
#_ = plt.boxplot([y1,y2,y3,y4])

g1 = dataset.avg_consumption[dataset['heating_fuel']=='gas']
g2 = dataset.avg_consumption[dataset['heating_fuel']=='elec']
g3 = dataset.avg_consumption[dataset['heating_fuel']=='lpg_oil']
g4 = dataset.avg_consumption[dataset['heating_fuel']=='other']
glabels = ['gas','electric','lpg oil','other']
#_  = plt.boxplot([g1,g2,g3,g4],labels = glabels)

t1 = dataset.avg_consumption[dataset['heating_temperature']=='below_18']
t2 = dataset.avg_consumption[dataset['heating_temperature']=='18_to_20']
t3 = dataset.avg_consumption[dataset['heating_temperature']=='above_20']
t4 = dataset.avg_consumption[dataset['heating_temperature']=='not_sure']
tlabels = ['Below 18','18 to 20','Above 20','Not sure']
#_ = plt.boxplot([t1,t2,t3,t4],labels = tlabels)

i1 = dataset.avg_consumption[dataset['wall_insulation']=='n']
i2 = dataset.avg_consumption[dataset['wall_insulation']=='y_cavity']
i3 = dataset.avg_consumption[dataset['wall_insulation']=='y_internal']
i4 = dataset.avg_consumption[dataset['wall_insulation']=='y_external']
i5 = dataset.avg_consumption[dataset['wall_insulation']=='not_sure']
ilabels = ['No','Cavity','Internal','External','Not sure']
#_ = plt.boxplot([i1,i2,i3,i4,i5],labels = ilabels)

l1 = dataset.avg_consumption[dataset['efficient_lighting_percentage']=='0_to_25']
l2 = dataset.avg_consumption[dataset['efficient_lighting_percentage']=='25_to_50']
l3 = dataset.avg_consumption[dataset['efficient_lighting_percentage']=='50_to_75']
l4 = dataset.avg_consumption[dataset['efficient_lighting_percentage']=='75_to_100']
llabels = ['0 to 25','25 to 50','50 to 75','75 to 100']
#plt.boxplot([l1,l2,l3,l4],labels = llabels)

plt.scatter(dataset.num_bedrooms,dataset.avg_consumption)
plt.show()