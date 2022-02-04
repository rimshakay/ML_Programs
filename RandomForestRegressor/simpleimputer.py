# from matplotlib.pyplot import axis
# from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

train_csv="data/titanic_data/train.csv"
test_file="data/titanic_data/test.csv"

train_data=pd.read_csv(train_csv)
test_data=pd.read_csv(test_file)

#SibSp and Parch does not help
columns=['Pclass','Age','Fare']

low_cardinality_cols = [cname for cname in train_data.columns if train_data[cname].nunique() < 10 and 
                        train_data[cname].dtype == "object"]

print(low_cardinality_cols)

train_Y=train_data.dropna(subset=low_cardinality_cols).Survived
train_X=train_data[columns+low_cardinality_cols]
test_X=test_data[columns+low_cardinality_cols]


label_X_train=train_X.copy()
label_X_test=test_X.copy()

label_X_train=label_X_train.dropna(subset=low_cardinality_cols)
print(label_X_train[low_cardinality_cols].isnull().sum())
ordinal_encoder=OrdinalEncoder()
label_X_train[low_cardinality_cols]=ordinal_encoder.fit_transform(label_X_train[low_cardinality_cols])
label_X_test[low_cardinality_cols]=ordinal_encoder.transform(label_X_test[low_cardinality_cols])


# train_Y=train_data.Survived
# train_X=train_data[columns]
# train_X.loc[train_X['Sex'] == 'female', 'Sex'] = 0
# train_X.loc[train_X['Sex'] == 'male', 'Sex'] = 1
# # test_X=test_data[columns]
# test_X.loc[test_X['Sex'] == 'female', 'Sex'] = 0
# test_X.loc[test_X['Sex'] == 'male', 'Sex'] = 1

imputer=SimpleImputer()
imputed_train_X=pd.DataFrame(imputer.fit_transform(label_X_train))
imputed_test_X=pd.DataFrame(imputer.transform(label_X_test))

imputed_train_X.columns=train_X.columns
imputed_test_X.columns=test_X.columns

RandomForestModel=RandomForestRegressor(random_state=1)
RandomForestModel.fit(imputed_train_X,train_Y)

Survival_predictions=RandomForestModel.predict(imputed_test_X)

list_of_tuples=list(zip(test_data['PassengerId'],Survival_predictions.round().astype(int)))
predictions_df=pd.DataFrame(list_of_tuples,columns=['PassengerId','Survived'])

predictions_df.to_csv('data/titanic_data/random_forest_predictions.csv',index=False)


