from forest_type.entity import config_entity , artifact_entity
from forest_type.exception import ForestException
from forest_type.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from forest_type import utils
import pandas as pd
import numpy as np
import os , sys

class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig, 
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
            try:
                self.data_transformation_config = data_transformation_config
                self.data_ingestion_artifact = data_ingestion_artifact
            except Exception as e:
                raise ForestException(e, sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            robust_scaler =  RobustScaler()
            pipeline =  Pipeline(steps=[
                     ('RobustScaler',robust_scaler)])
                
            return pipeline    
        except Exception as e:
            raise ForestException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file 
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #Selecting input features for train and test dataframe
            input_feature_train_df = train_df.drop(train_df.iloc[:,-1:], axis=1)
            input_feature_test_df = test_df.drop(test_df.iloc[:,-1:], axis=1)

            #Selecting target feature for train and test dataframe
            target_feature_train_df = train_df.iloc[:,-1:]
            target_feature_test_df = test_df.iloc[:,-1:]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)


            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            #Transforming input feature
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            smt = SMOTE()
            
            logging.info(f"Before resampling in Training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_df}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in Training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_df}")

            logging.info(f"Before resampling in Testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_df}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in Testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_df}")

            #Target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                                obj=transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)                    


            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )
            logging.info(f"Data transformation object: {data_transformation_artifact}")                    
            return data_transformation_artifact 
        except Exception as e:
            raise ForestException(e, sys)        