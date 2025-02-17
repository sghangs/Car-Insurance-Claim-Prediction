import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from claimprediction.constant.training_pipeline import (
    TARGET_COLUMN,
    ONE_HOT_ENCODER_FEATURES,
    ORDINAL_ENCODER_FEATURES,
    TARGET_ENCODER_FEATURES,
    UNIQUE_ID_COLUMN
)
from claimprediction.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from claimprediction.entity.config_entity import DataTransformationConfig,DataValidationConfig
from claimprediction.exception.exception import CarInsuranceException
from claimprediction.logging.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,TargetEncoder
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from claimprediction.utils.main_utils.utils import save_object,save_numpy_array_data



class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise CarInsuranceException(e)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CarInsuranceException(e)

    def remove_outliers(self,dataframe):
        try:
            #Remove outliers from age_of_car feature
            IQR=dataframe['age_of_car'].quantile(0.75)-dataframe['age_of_car'].quantile(0.25)
            lower_boundary=dataframe['age_of_car'].quantile(0.25)-IQR*1.5
            upper_boundary=dataframe['age_of_car'].quantile(0.75)+IQR*1.5
            dataframe['age_of_car']=dataframe['age_of_car'].mask(
                dataframe['age_of_car'] > upper_boundary,upper_boundary)
            dataframe['age_of_car']=dataframe['age_of_car'].mask(
                dataframe['age_of_car'] < lower_boundary,lower_boundary)
            
            #Remove outliers from age_of_policyholder feature
            IQR=dataframe['age_of_policyholder'].quantile(0.75)-dataframe['age_of_policyholder'].quantile(0.25)
            lower_boundary=dataframe['age_of_policyholder'].quantile(0.25)-IQR*1.5
            upper_boundary=dataframe['age_of_policyholder'].quantile(0.75)+IQR*1.5
            dataframe['age_of_policyholder']=dataframe['age_of_policyholder'].mask(
                dataframe['age_of_policyholder'] > upper_boundary,upper_boundary)
            dataframe['age_of_policyholder']=dataframe['age_of_policyholder'].mask(
                dataframe['age_of_policyholder'] < lower_boundary,lower_boundary)

            return dataframe
        except Exception as e:
            raise CarInsuranceException(e)

    def get_data_transformer_object(self) -> Pipeline:
        '''
        This function is responsible for data transformation
        '''
        try:
            trf1=ColumnTransformer([
                ("One_hot_encoder",OneHotEncoder(handle_unknown="ignore"),ONE_HOT_ENCODER_FEATURES),
                ("Ordinal_encoder",OrdinalEncoder(),ORDINAL_ENCODER_FEATURES),
                ("Target_encoder",TargetEncoder(),TARGET_ENCODER_FEATURES),
                ("Power_transformer",PowerTransformer(method='yeo-johnson'),["age_of_car"])
            ],remainder="passthrough")

            trf2=ColumnTransformer([
                ("scaler",StandardScaler(),slice(0,41))
            ])

            preprocessor = Pipeline([
                ("encoding_transformer",trf1),
                ("scaler_transformer",trf2)
            ])

            logging.info("Encoding, transformation and scaling is completed")

            return preprocessor
        except Exception as e:
            raise CarInsuranceException(e)

    def handle_imbalance_class(self,X_train,y_train):
        try:
            logging.info("Before oversampling, counts of 1 : {}".format(sum(y_train==1)))
            logging.info("Before oversampling, counts of 0 : {}".format(sum(y_train==0)))

            smote_object=SMOTE()
            X_train_sm,y_train_sm=smote_object.fit_resample(X_train,y_train)
            
            logging.info("After oversampling, counts of 1 : {}".format(sum(y_train_sm==1)))
            logging.info("After oversampling, counts of 0 : {}".format(sum(y_train_sm==0)))

            return X_train_sm,y_train_sm

        except Exception as e:
            raise CarInsuranceException(e)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")

            # Read data from csv file into dataframe
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            #Creating dependent and Independent features
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]

            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]

            # Remove unique id feature 
            input_feature_train_df=input_feature_train_df.drop(columns=[UNIQUE_ID_COLUMN],axis=1)
            input_feature_test_df=input_feature_test_df.drop(columns=[UNIQUE_ID_COLUMN],axis=1)

            # Remove Outliers from Numerical Columns (age of car and age of policyholder)
            input_feature_train_df=self.remove_outliers(input_feature_train_df)
            input_feature_test_df=self.remove_outliers(input_feature_test_df)

            # Encode, Transform and Scale features
            preprocessor=self.get_data_transformer_object()
            preprocessor_object=preprocessor.fit(input_feature_train_df,target_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature=preprocessor_object.transform(input_feature_test_df)

            logging.info("Input train data shape : ",transformed_input_train_feature.shape)
            logging.info("Input test data shape : ",transformed_input_test_feature.shape)

            # Handle Imbalance dataset (Using SMOTE)

            input_train_feature_sm,target_train_feature_sm=self.handle_imbalance_class(
                transformed_input_train_feature,np.array(target_feature_train_df)
            )

            # Concatenate input feature with target feature
            train_arr=np.c_[input_train_feature_sm,np.array(target_train_feature_sm)]
            test_arr=np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]

            # Save train and test numpy array into numpy object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            
            # Save preprocessor object 
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            #prepare Artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact


        except Exception as e:
                raise CarInsuranceException(e)