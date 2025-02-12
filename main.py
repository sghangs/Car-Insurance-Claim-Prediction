from claimprediction.exception.exception import CarInsuranceException
from claimprediction.logging.logger import logging
from claimprediction.components.data_ingestion import DataIngestion
from claimprediction.entity.artifact_entity import DataIngestionArtifact
from claimprediction.entity.config_entity import DataIngestionConfig
from claimprediction.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=="__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)

    except Exception as e:
        raise CarInsuranceException(e)

