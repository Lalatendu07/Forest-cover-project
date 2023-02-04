import os,sys
from forest_type.logger import logging
from forest_type.exception import ForestException
from forest_type.utils import get_collection_as_DataFrame
from forest_type.entity import config_entity 
from forest_type.Component.data_ingestion import DataIngestion
from forest_type.Component.data_validation import DataValidation
from forest_type.Component.data_transformation import DataTransformation
from forest_type.Component.model_trainer import ModelTrainer


if __name__=='__main__':
     try:
        #Data ingestion
         training_pipeline_config = config_entity.TrainingPipelineConfig()
         data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
         print(data_ingestion_config.to_dict())
         data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
         data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        #Data validation
         data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
         data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)

         data_validation_artifact = data_validation.initiate_data_validation()

        #Data transformation
         data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
         data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                   data_ingestion_artifact=data_ingestion_artifact)
         data_transformation_artifact = data_transformation.initiate_data_transformation() 

        #Model training
         model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
         model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
         model_trainer_artifact = model_trainer.initiate_model_trainer()

     except Exception as e :
          print(e)     
