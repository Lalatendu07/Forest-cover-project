from forest_type.pipeline.Batch_prediction import start_batch_prediction


file_path = '/config/workspace/covtype.data.gz'

if __name__=='__main__':
     try:
          output_file=start_batch_prediction(input_file_path=file_path)
          print(output_file)
     except Exception as e :
          print(e)     
