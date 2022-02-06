# from matplotlib.pyplot import axis
# from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

class myRandomForest:

    def __init__(self,train_file,test_file,prediction_column,prediction_key=None):
        self.train_file=train_file
        self.test_file=test_file
        self.prediction_column=prediction_column
        self.prediction_key=prediction_key

    def startPipeline(self):

        train_csv=self.train_file
        test_file=self.test_file
        prediction_column=self.prediction_column
        prediction_key=self.prediction_key

        train_data=pd.read_csv(train_csv)
        test_data=pd.read_csv(test_file)

        #SibSp and Parch does not help
        # columns=['Pclass','Age','Fare']

        train_Y=train_data[prediction_column]
        train_X=train_data.drop([prediction_column],axis=1)

        numerical_cols=[cname for cname in train_X.columns if
                                train_X[cname].dtype in ['int64','float64']]
        low_cardinality_cols = [cname for cname in train_X.columns if train_X[cname].nunique() < 10 and 
                                train_X[cname].dtype == "object"]
        high_cardinality_cols= [cname for cname in train_X.columns if train_X[cname].nunique() > 10 and 
                                train_X[cname].dtype == "object"]


        train_X=train_data[numerical_cols+low_cardinality_cols]
        test_X=test_data[numerical_cols+low_cardinality_cols]

        label_X_train=train_X.copy()
        label_X_test=test_X.copy()

        # label_X_train=label_X_train.drop(high_cardinality_cols,axis=1)

        numerical_transformer=Pipeline([('imputer',SimpleImputer(strategy='median'))])

        categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('ordinal_encoder',OrdinalEncoder(handle_unknown='ignore'))])

        preprocessor=ColumnTransformer(transformers=[
            ('num',numerical_transformer,numerical_cols),
            ('cat',categorical_transformer,low_cardinality_cols)
        ])

        model=RandomForestRegressor(random_state=1)


        main_pipeline=Pipeline([('preprocessor',preprocessor),
        ('model',model)])

        main_pipeline.fit(train_X,train_Y)

        Survival_predictions=main_pipeline.predict(test_X)

        list_of_tuples=list(zip(test_data[prediction_key],Survival_predictions.round().astype(int)))
        predictions_df=pd.DataFrame(list_of_tuples,columns=[prediction_key,prediction_column])

        folder_path="/".join(train_csv.split("/")[:-1])
        predictions_df.to_csv(folder_path+'/random_forest_predictions.csv',index=False)


call1=myRandomForest("data/titanic_data/train.csv","data/titanic_data/test.csv","Survived","PassengerId")

call1.startPipeline()