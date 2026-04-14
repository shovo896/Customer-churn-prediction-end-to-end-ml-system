import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 


RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def load_and_clean(path:str) -> pd.DataFrame:
    df=pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    print(f"Data loaded and cleaned successfully with shape: {df.shape}")
    return df

def engineer_features(df:pd.DataFrame) -> pd.DataFrame:
    df['tenure_group']=pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=['0-12', '13-24', '25-48', '49-60', '60+'])
    df['charge_ratio']=df['MonthlyCharges']/(df['TotalCharges'] + 1)
    print(f"Feature engineering:{df.shape}")
    return df