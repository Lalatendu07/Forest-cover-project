import pymongo
import pandas as pd
import json
from forest_type.config import mongo_client



DATA_FILE_PATH = '/config/workspace/covtype.data.gz'
DATABASE_NAME = 'forest_cover'
COLLECTION_NAME = 'cover_type_data'

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE_PATH)
    print(f'Rows and columns: {df.shape}')

    #Convert dataframe to json so that we can dump these record into mongodb
    df.reset_index(drop=True,inplace=True)
    
    #Taking sample dataset
    data=df.sample(40000)

    json_record =list(json.loads(data.T.to_json()).values())

    print(json_record[0])

    #insert converted json record in mongodb 
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)