from forest_type.pipeline.TrainingPipeline import start_training_pipeline 


file_path = '/config/workspace/covtype.data.gz'

if __name__ == "__main__" :
    try:
        start_training_pipeline()
    except Exception as e:
        print(e)    
