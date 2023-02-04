import os,sys
from forest_type.exception import ForestException
from forest_type.logger import logging
from datetime import datetime

FILE_NAME = 'forest.csv'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"

class TrainingPipelineConfig:
    def __init__(self):
        
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        

class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = 'forest_cover'
            self.collection_name = 'cover_type_data'
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,'data_ingestion')
            self.features_store_file_path = os.path.join(self.data_ingestion_dir,'features_store',FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise ForestException(e, sys)
            
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e :
            raise ForestException(e, sys)        


class DataValidationConfig:

     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir, 'report.yaml')
        self.base_file_path = os.path.join("covtype.data.gz")


class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation_dir")
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path = os.path.join(self.data_transformation_dir,"transformed","TRAIN_FILE_NAME")
        self.transformed_test_path = os.path.join(self.data_transformation_dir,"transformed","TEST_FILE_PATH")
        self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)



class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model", MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_thres = 0.1


class ModelEvaluationConfig:...
class ModelPusherConfig:...