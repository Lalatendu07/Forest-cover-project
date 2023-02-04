from forest_type.entity import config_entity , artifact_entity
from forest_type.exception import ForestException
from forest_type.logger import logging
import os , sys
from xgboost import XGBClassifier
from forest_type import utils
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV



class ModelTrainer:


    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                  data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact 
        except Exception as e:
            raise ForestException(e, sys)

    #Hyperparameter tuning using GridSearchCV
    #def model_tuner(self,x,y):
    #    try:
     #       param_grid_xgboost = {'learning_rate': [0.5, 0.1, 0.01, 0.001],'max_depth': [3, 5, 10, 20],'n_estimators': [10, 50, 100, 200]}
      #      grid= GridSearchCV(XGBClassifier(objective='multi:softprob'),param_grid_xgboost, verbose=3,cv=5,n_jobs=-1)
       #     grid.fit(x,y)
        #    learning_rate = grid.best_params_['learning_rate']
        #    max_depth = grid.best_params_['max_depth']
        #    n_estimators = grid.best_params_['n_estimators']
        #    xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
         #   return xgb
                              
        except Exception as e:
            raise ForestException(e, sys)


    def train_model(self,x,y):
        try:
            xgb_clf = XGBClassifier()#self.model_tuner(x=x,y=y)
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise ForestException(e, sys)    

    def initiate_model_trainer(self,) -> artifact_entity.ModelTrainerArtifact:

        try:
            logging.info("Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting input and target feature from both train and test array.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info("Train the model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train , y_pred=yhat_train , average='micro')

            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test , y_pred=yhat_test , average='micro')

            logging.info(f"Train score: {f1_train_score} and test score: {f1_test_score}")
            #Check for overfitting or underfitting or expected score
            logging.info("Checking if our model is underfitted or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score} model actual score: {f1_test_score}")
              
            logging.info(f"Checking if our model is overfiiting or not")  
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_thres:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_thres}")    
            
            #Save the trained model
            logging.info(f"Saving the model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #Prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path = self.model_trainer_config.model_path,
                f1_train_score = f1_train_score,
                f1_test_score = f1_test_score
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise ForestException(e, sys)                