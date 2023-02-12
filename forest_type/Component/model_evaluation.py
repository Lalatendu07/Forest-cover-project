from forest_type.predictor import ModelResolver
from forest_type.entity import config_entity,artifact_entity
from forest_type.utils import load_object
from forest_type.exception import ForestException
from forest_type.logger import logging
from sklearn.metrics import f1_score
import pandas as pd
import os,sys


class ModelEvaluation:

    def __init__(self,
            model_eval_config:config_entity.ModelEvaluationConfig,
            data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
            data_transformation_artifact:artifact_entity.DataTransformationArtifact,
            model_trainer_artifact:artifact_entity.ModelTrainerArtifact
                 ):

            try:
                logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
                self.model_eval_config=model_eval_config
                self.data_ingestion_artifact=data_ingestion_artifact
                self.data_transformation_artifact=data_transformation_artifact
                self.model_trainer_artifact=model_trainer_artifact
                self.model_resolver=ModelResolver()

            except Exception as e:
                raise ForestException(e, sys)   


    def initiate_model_evaluation(self)-> artifact_entity.ModelEvaluationArtifact:
        try:
            # if saved model folder has model then we will compare
            # Which model is best trained or the model from saved model folder

            logging.info("if saved model folder has model then we will compare"
            "Which model is best trained or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model evalution artifact: {model_eval_artifact}")
                return model_eval_artifact
           

            # Finding the location of transformer , model and target encoder
            logging.info("Finding the location of transformer , model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            #Previous trained objects
            logging.info("Previous trained objects of transformer , model and target encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            #Currently trained model object
            logging.info("Currently trained model object of transformer , model and target encoder")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df.iloc[:,-1:]
            y_true = target_encoder.transform(target_df)

            #accuracy using previous trained model 
            logging.info("accuracy using previous trained model")
            input_feature_name = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred,average='micro')
            logging.info(f"accuracy using previous trained model: {previous_model_score}")

            #accuracy using current trained model 
            logging.info("accuracy using current trained model")
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred,average='micro')
            logging.info(f"accuracy using current trained model: {current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info("Current trained model is not better than previous trained model")
                raise Exception("Current trained model is not better than previous trained model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
            improved_accuracy=current_model_score-previous_model_score)

            logging.info(f"model eval artifact: {improved_accuracy}") 
            return model   


        except Exception as e:
            raise ForestException(e, sys)    
                     