import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

data_set = pd.read_csv('HIS/insurance.csv')

print(data_set.head())
print(data_set.info())


data_set['region'] = data_set['region'].astype('category').cat.codes
data_set['smoker'] = data_set['smoker'].map({'yes': 1, 'no': 0})
data_set['sex'] = data_set['sex'].map({'male': 1, 'female': 0})
print(data_set.corr()['charges'].sort_values())


# # make a copy of the original data and make smoker yes and no to 1 and 0 and sex male and female to 1 and 0
# data = data_set.copy()
# data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
# data['sex'] = data['sex'].map({'male': 1, 'female': 0})
