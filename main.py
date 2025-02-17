from claimprediction.exception.exception import CarInsuranceException
from claimprediction.logging.logger import logging
from claimprediction.components.data_ingestion import DataIngestion
from claimprediction.components.data_validation import DataValidation
from claimprediction.components.data_transformation import DataTransformation
from claimprediction.components.feature_selection import FeatureSelection
from claimprediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    FeatureSelectionArtifact
)
from claimprediction.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    FeatureSelectionConfig)
import sys

if __name__=="__main__":
    try:
        ## Data Ingestion

        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)

        ## Data Validation

        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("INitiate data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data validation completed")

        ## Data Transformation

        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("INitiate data Transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("data transformation completed")

        ## Feature Selection

        feature_selection_config=FeatureSelectionConfig(training_pipeline_config)
        feature_selection=FeatureSelection(data_transformation_artifact,feature_selection_config)
        logging.info("Initiate Feature Selection")
        feature_selection_artifact=feature_selection.initiate_feature_selection()
        logging.info("Feature Selection completed")

    except Exception as e:
        raise CarInsuranceException(e)

