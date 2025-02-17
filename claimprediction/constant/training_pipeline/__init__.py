import numpy as np
import pandas as pd
import os
import sys

"""
Defining common constant variable for training pipeline
"""
TARGET_COLUMN = "is_claim"
UNIQUE_ID_COLUMN="policy_id"
PIPELINE_NAME: str = "ClaimPrediction"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "CarInsuranceData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME="model.pkl"

ONE_HOT_ENCODER_FEATURES=[
                "make",
                "segment",
                "fuel_type",
                "is_esc",
                "is_adjustable_steering",
                "is_tpms",
                "is_parking_sensors",
                "is_parking_camera",
                "rear_brakes_type",
                "transmission_type",
                "steering_type",
                "is_front_fog_lights",
                "is_rear_window_wiper",
                "is_rear_window_washer",
                "is_rear_window_defogger",
                "is_brake_assist",
                "is_power_door_locks",
                "is_central_locking",
                "is_power_steering",
                "is_driver_seat_height_adjustable",
                "is_day_night_rear_view_mirror",
                "is_ecw",
                "is_speed_alert"
            ]

ORDINAL_ENCODER_FEATURES=[
    "airbags",
    "displacement",
    "cylinder",
    "gear_box",
    "turning_radius",
    "length",
    "width",
    "height",
    "gross_weight",
    "ncap_rating",
    "population_density"
]

TARGET_ENCODER_FEATURES=[
    "area_cluster",
    "model",
    "max_torque",
    "max_power",
    "engine_type"
]



"""
Data Ingestion related constant start with DATA_INGESTION var name
"""
DATA_INGESTION_COLLECTION_NAME: str = "CarInsuranceData"
DATA_INGESTION_DATABASE_NAME: str = "SGHANGS"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION var name
"""
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Feature Selection related constant start with FEATURE_SELECTION VAR NAME
"""
FEATURE_SELECTION_DIR_NAME: str = "feature_selection"
FEATURE_SELECTION_DATA_DIR: str = "selected_feature"
FEATURE_SELECTION_TRAIN_FILE_PATH: str = "train.npy"


