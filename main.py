import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.utils import get_collection_as_DataFrame
from sensor.entity.config_entity import DataIngestionConfig , TrainingPipelineConfig
from sensor.Component.data_ingestion import DataIngestion

if __name__=='__main__':
     try:
         training_pipeline_config = TrainingPipelineConfig()
         data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
         print(data_ingestion_config.to_dict())
         data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
         print(data_ingestion.initiate_data_ingestion())

     except Exception as e :
          print(e)     
