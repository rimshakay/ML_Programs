from matplotlib.pyplot import axis
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_csv="DecisionTreeRegressor/data/train.csv"
test_file="DecisionTreeRegressor/data/test.csv"

train_data=pd.read_csv(train_csv)
test_data=pd.read_csv(test_file)

train_Y=train_data.Survived
train_X=train_data.drop(['Survived'],axis=1)



