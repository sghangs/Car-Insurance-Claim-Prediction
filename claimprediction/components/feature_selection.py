import sys
import os
import numpy as np
import pandas as pd

from claimprediction.exception.exception import CarInsuranceException
from claimprediction.logging.logger import logging
from claimprediction.entity.artifact_entity import (
    DataTransformationArtifact,
    FeatureSelectionArtifact
)
from claimprediction.entity.config_entity import FeatureSelectionConfig
from claimprediction.utils.main_utils.utils import (
    save_numpy_array_data,
    load_numpy_array_data
)
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

class FeatureSelection:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                feature_selection_config:FeatureSelectionConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.feature_selection_config=feature_selection_config
        except Exception as e:
            raise CarInsuranceException(e)
    
    def get_best_features(self,X_train,y_train):
        try:
            # Used RandomForest as base model for feature selection
            rf_object=RandomForestClassifier()
            rf_object.fit(X_train,y_train)

            # Used Backward Elimination feature selection technique
            be_object=SequentialFeatureSelector(
                rf_object,
                k_features="best",
                forward=False,
                scoring="f1",
                n_jobs=-1
            )
                
            be_object.fit(X_train,y_train)

            # Get indices of best features list
            best_features=list(be_object.k_feature_idx_)

            rows,columns=X_train.shape

            logging.info("Number of features before Feature Selection : ",columns)
            logging.info("Number of features after Feature Selection : ",len(best_features))
            logging.info("Average score of selected subset : ",be_object.k_score_)

            return best_features

        except Exception as e:
            raise CarInsuranceException(e)

    def initiate_feature_selection(self):
        try:
            logging.info("Feature Selection is starting")
            start_time=datetime.now()
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_arr = load_numpy_array_data(train_file_path)

            x_train, y_train = (
                train_arr[:, :-1],
                train_arr[:, -1]
            )

            # Get indices of best features using backward elimination technique
            best_features_list=self.get_best_features(x_train,y_train)

            # Get input and target set of selected features from train data
            X_train_final,y_train_final=x_train[:,best_features_list],y_train

            # Concatenate input and target feature
            train_arr_final=np.c_[X_train_final,y_train_final]

            # Save final train data as numpy array
            save_numpy_array_data(
                self.feature_selection_config.feature_selected_train_file_path,
                array=train_arr_final)

            logging.info("Feature Selection is completed")

            end_time=datetime.now()
            time_taken=end_time - start_time
            logging.info("Time taken by Feature selection process : ",time_taken)
            
            logging.info("Input train data shape : ",X_train_final.shape)
            logging.info("Target train data shape : ",y_train_final.shape)

            # Prepare Artifacts
            feature_selection_artifact=FeatureSelectionArtifact(
                feature_selected_train_file_path=self.feature_selection_config.feature_selected_train_file_path,
            )

            return feature_selection_artifact

        except Exception as e:
            raise CarInsuranceException(e)

