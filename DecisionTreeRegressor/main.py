from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

current_population_path="DecisionTreeRegressor/archive/CurrentPopulationSurvey.csv"
income_dynamic_path="DecisionTreeRegressor/archive/PanelStudyIncomeDynamics.csv"

population_df=pd.read_csv(current_population_path)
income_df=pd.read_csv(income_dynamic_path)

train_X, val_X, train_y, val_y = train_test_split()

