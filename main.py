from claimprediction.exception.exception import CarInsuranceException
from claimprediction.logging.logger import logging
from claimprediction.components.data_ingestion import DataIngestion
from claimprediction.components.data_validation import DataValidation
from claimprediction.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from claimprediction.entity.config_entity import DataIngestionConfig,DataValidationConfig
from claimprediction.entity.config_entity import TrainingPipelineConfig
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

    except Exception as e:
        raise CarInsuranceException(e)

