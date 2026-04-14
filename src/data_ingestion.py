import os 
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
Raw_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
def load_data():
    df=pd.read_csv(Raw_DATA_PATH)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df

def basic_info(df:pd.DataFrame):
    print("Basic Information about the dataset:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
