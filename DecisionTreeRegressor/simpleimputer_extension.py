# from matplotlib.pyplot import axis
# from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

train_csv="data/titanic_data/train.csv"
test_file="data/titanic_data/test.csv"

train_data=pd.read_csv(train_csv)
test_data=pd.read_csv(test_file)

columns=['Pclass','Age', 'Sex', 'Fare']
train_Y=train_data.Survived
train_X=train_data[columns]
train_X.loc[train_X['Sex'] == 'female', 'Sex'] = 0
train_X.loc[train_X['Sex'] == 'male', 'Sex'] = 1
test_X=test_data[columns]
test_X.loc[test_X['Sex'] == 'female', 'Sex'] = 0
test_X.loc[test_X['Sex'] == 'male', 'Sex'] = 1

cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]
                     
train_X_plus = train_X.copy()
test_X_plus = test_X.copy()

for col in cols_with_missing:
    train_X_plus[col+ '_was_missing'] = train_X_plus[col].isnull()
    test_X_plus[col+ '_was_missing'] = test_X_plus[col].isnull()

imputer=SimpleImputer()
imputed_train_X=pd.DataFrame(imputer.fit_transform(train_X))
imputed_test_X=pd.DataFrame(imputer.transform(test_X))

imputed_train_X.columns=train_X.columns
imputed_test_X.columns=test_X.columns

DecisionTreeModel=DecisionTreeRegressor()
DecisionTreeModel.fit(imputed_train_X,train_Y)

Survival_predictions=DecisionTreeModel.predict(imputed_test_X)

list_of_tuples=list(zip(test_data['PassengerId'],Survival_predictions.round().astype(int)))
predictions_df=pd.DataFrame(list_of_tuples,columns=['PassengerId','Survived'])
# predictions_df.reset_index(drop=True, inplace=True)

predictions_df.to_csv('data/titanic_data/decision_tree_predictions.csv',index=False)


