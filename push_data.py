import os
import sys
import json
import pandas as pd
import numpy as np
import pymongo
from claimprediction.logging.logger import logging
from claimprediction.exception.exception import CarInsuranceException

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class CarInsuranceExtract:
    def __init__(self) -> None:
        pass

    def csv_to_json_converter(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=json.loads(data.to_json(orient="records"))
            return records

        except Exception as e:
            raise CarInsuranceException(e)
    
    def insert_data_mongodb(self,database,collection,records):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongodb_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongodb_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)

            return len(self.records)
        except Exception as e:
            raise CarInsuranceException(e)

if __name__=="__main__":
    FILE_PATH="claimprediction_data/car_insurance_data.csv"
    DATABASE="SGHANGS"
    Collection="CarInsuranceData"
    obj=CarInsuranceExtract()
    records=obj.csv_to_json_converter(FILE_PATH)
    no_of_records=obj.insert_data_mongodb(DATABASE,Collection,records)
    print("Number of records insert : ",no_of_records)
