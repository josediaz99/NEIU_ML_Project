# ETL pipeline for machine learning
import os
import pandas as pd
from datetime import datetime
import sqlite3


def extract(folder:str, logfile='none') -> pd.DataFrame:
    """
    Extract takes in a folder of json,csv's, and parquet
    files and saves them as a pandas dataframe 
    """
    dataframes = []
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        if file.endswith('.csv'):
            df = pd.read_csv(path)
        elif file.endswith('parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_json(path, lines=True)
        log(logfile=logfile,message=f"loaded {file} of shape : {df.shape}")
        dataframes.append(df)
    
    # concatenates all dataframes to one
    data = pd.concat(dataframes, ignore_index=True)
    return data

def transform(df:pd.DataFrame, logfile='none'):
    """
    this function takes in data which needs to be prepared for our machine learning  model
    
    Args:
        df (pd.DataFrame): uncleaned dataframe containing 78 features of network data

    Returns:
        pd.DataFrame (x,y) : returning two dataframes which will contain the cleaned data and labels
    """
    log(logfile,'cleaning data...')
    df.columns = df.columns.str.lstrip()
    df = df.rename(columns={'Label':'attack'})
    df['attack'] = df['attack'].apply(lambda x: 1 if "dos" in str(x).lower() else 0 )
    
    log(logfile,'feature engineering...')
    df['f_b_total_packet_ratio'] =  df['Total Backward Packets'] / (1 + df['Total Length of Fwd Packets'])
    
    log(logfile,'filtering data...')
    wanted = ['f_b_total_packet_ratio', 'Fwd Packet Length Max', 'Bwd Packet Length Std', 'Flow IAT Mean', 'act_data_pkt_fwd']
    
    return df[wanted].copy(), df['attack'].copy()
        
def load(data, dbname:str, tableName:str) -> None:
    """
    load takes in the data  which we want to store, the database name, and table name
    and stores this information in an sqlite database

    Args:
        data (_type_): processed dataframe only containing important features for our machine learning to use
        dbname (str): this will be the database we are trying to save  to 
        tableName (str): this is the table name we are trying to save our information to
    """
    sqlconnection = sqlite3.connect(dbname)
    data.to_sql(tableName, sqlconnection, if_exists='replace', index=False)
    
def ETL(folder:str, dbname:str,) -> None:
    feature_table = 'features'
    labels_table = 'labels'
    logfile = 'ETL_log'
    
    log(logfile=logfile, message='ETL pipeline started')
    
    log(logfile=logfile,message='extracting...')
    df = extract(folder=folder,logfile=logfile)
    log(logfile=logfile,message='extraction complete!')
    
    log(logfile=logfile,message='transforming data...')
    x,y = transform(df=df,logfile=logfile)
    log(logfile=logfile,message='transformation complete!')
    
    log(logfile=logfile,message='loading data...')
    
    log(logfile=logfile,message='loading data features...')
    load(data=x, dbname=dbname, tableName=feature_table)  
    log(logfile=logfile,message='loading data features complete!')
    
    log(logfile=logfile,message='loading data labels...')
    load(data=y, dbname=dbname, tableName=labels_table)
    log(logfile=logfile,message='loading data labels complete!')
     
    log(logfile=logfile,message='loading complete!')
    
    log(logfile=logfile,message='ETL pipeline complete!')
    
def log(logfile:str, message:str) -> None:
    if logfile != 'none':
        datatime_format = '%Y-%m-%d-%H:%M:%S.%f, '
        datetimestamp = datetime.now().strftime(datatime_format)
        with open(logfile, 'a') as file:
            file.write(datetimestamp + message + '\n')
    else:
        print(message)
        
if __name__ == "__main__":
    folder = "data/"
    dbname = 'machine_learning_data'
    ETL(folder=folder, dbname=dbname)
    
    
