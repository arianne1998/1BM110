from pprint import pprint

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

#import csv files into dataframes
info_df = pd.read_csv('Datasets/info.csv')
meter_consumption_df = pd.read_csv('Datasets/meter_consumption.csv')
weather_avg_df = pd.read_csv('Datasets/weather_avg.csv')
weather_max_df = pd.read_csv('Datasets/weather_max.csv')
weather_min_df = pd.read_csv('Datasets/weather_min.csv')

