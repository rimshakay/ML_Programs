# from matplotlib.pyplot import axis
# from sklearn.model_selection import train_test_split
# from openerp import api,models,fields
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import datetime

# class myRandomForest(models.Model):
class myRandomForest:

    def __init__(self,imputer,train_file,test_file,prediction_column,prediction_key=None):
        self.train_file=train_file
        self.test_file=test_file
        self.prediction_column=prediction_column
        self.prediction_key=prediction_key
        self.imputer=imputer

    def get_info(self):
        train_csv=self.train_file
        prediction_column=self.prediction_column

        train_data=pd.read_csv(train_csv)

        train_Y=train_data[prediction_column]
        train_X=train_data.drop([prediction_column],axis=1)

        numerical_cols=[cname for cname in train_X.columns if
                                train_X[cname].dtype in ['int64','float64']]
        print("Numerical columns:  "+",".join(numerical_cols))

        low_cardinality_cols = [cname for cname in train_X.columns if train_X[cname].nunique() < 8 and 
                                train_X[cname].dtype == "object"]
        print("Low Cardinality columns:  "+",".join(low_cardinality_cols))

        high_cardinality_cols= [cname for cname in train_X.columns if train_X[cname].nunique() > 7 and 
                                train_X[cname].dtype == "object"]
        print("High Cardinality columns:  "+",".join(high_cardinality_cols))

        return train_Y,train_X,numerical_cols,low_cardinality_cols,high_cardinality_cols

    def startPipeline(self):
        train_Y,train_X,numerical_cols,low_cardinality_cols,high_cardinality_cols=self.get_info()

        train_csv=self.train_file
        test_file=self.test_file
        prediction_column=self.prediction_column
        prediction_key=self.prediction_key

        test_data=pd.read_csv(test_file)
        print("read test csv")

        train_X=train_X[numerical_cols+low_cardinality_cols+high_cardinality_cols]
        test_X=test_data[numerical_cols+low_cardinality_cols+high_cardinality_cols]

        numerical_transformer=Pipeline([('imputer',SimpleImputer(strategy='median'))])
        ordinal_categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('ordinal_encoder',OrdinalEncoder(handle_unknown='ignore'))])
        onehot_categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])

        preprocessor=ColumnTransformer(transformers=[
            ('num',numerical_transformer,numerical_cols),
            ('ordinal_cat',ordinal_categorical_transformer,high_cardinality_cols),
            ('onehot_cat',onehot_categorical_transformer,low_cardinality_cols)
        ])

        model=RandomForestRegressor(n_estimators=100,random_state=1)
        print("creating random model")

        main_pipeline=Pipeline([('preprocessor',preprocessor),
        ('model',model)])

        print("fitting model")
        main_pipeline.fit(train_X,train_Y)
        
        print("making predictions")
        predictions=main_pipeline.predict(test_X)

        # print("Do you want the predictions rounded off to the nearest integer?(y/n)")
        # ans=input()

        # if ans=='y':
        predictions=predictions.round().astype(int)

        list_of_tuples=list(zip(test_data[prediction_key],predictions))
        predictions_df=pd.DataFrame(list_of_tuples,columns=[prediction_key,prediction_column])

        folder_path="/".join(train_csv.split("/")[:-1])
        predictions_df.to_csv(folder_path+'/random_forest_predictions.csv',index=False)

        return predictions_df

#Titanic competition
def checkTitanic(imputer):
    dt_start = datetime.datetime.now().strftime("%H:%M:%S")
    print("dt_start",dt_start)
    call1=myRandomForest(imputer,"data/titanic_data/train.csv","data/titanic_data/test.csv","Survived","PassengerId")
    df=call1.startPipeline()
    dt_end = datetime.datetime.now().strftime("%H:%M:%S")
    print(dt_end)

    return df

#Store sales competition
# @api.model
def checkSales():
    dt_start = datetime.datetime.now().strftime("%H:%M:%S")
    print("dt_start",dt_start)
    call1=myRandomForest("data/store-sales-time-series-forecasting/train.csv","data/store-sales-time-series-forecasting/test.csv","sales","id")
    df=call1.startPipeline()
    dt_end = datetime.datetime.now().strftime("%H:%M:%S")
    print(dt_end)

    return df
