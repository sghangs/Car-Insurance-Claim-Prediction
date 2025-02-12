import numpy as np
import pandas as pd
import os
import sys

"""
Defining common constant variable for training pipeline
"""
TARGET_COLUMN = "is_claim"
PIPELINE_NAME: str = "ClaimPrediction"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "CarInsuranceData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME="model.pkl"


"""
Data Ingestion related constant start with DATA_INGESTION var name
"""
DATA_INGESTION_COLLECTION_NAME: str = "CarInsuranceData"
DATA_INGESTION_DATABASE_NAME: str = "SGHANGS"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2