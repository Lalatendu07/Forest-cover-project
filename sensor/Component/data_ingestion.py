from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os , sys
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys) 

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Exporting collection as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_DataFrame(
                database_name=self.data_ingestion_config.database_name,
                 collection_name=self.data_ingestion_config.collection_name)

            logging.info("Create feature store folder if not available")     
            #Create features store directory
            features_store_dir = os.path.dirname(self.data_ingestion_config.features_store_file_path)   
            os.makedirs(features_store_dir,exist_ok=True)  
            
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.features_store_file_path,index=False,header=True)
            
            logging.info("Spliting data into train and test")
            #Spliting dataset into train and test
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size ,random_state=42)
            
            logging.info("Create dataset directory folder if not available")
            #Create dataset directory
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)
            
            logging.info("Saving dataframe to feature store")
            #Save dataframe into feature store folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            #Prepare artifact
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                features_store_file_path=self.data_ingestion_config.features_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)           