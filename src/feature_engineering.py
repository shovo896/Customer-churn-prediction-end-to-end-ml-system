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


def encode_and_scale(df:pd.DataFrame) -> pd.DataFrame:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    binary_cols =['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling','gender','paperLessBilling']
    for col in binary_cols:
        df[col]=LabelEncoder().fit_transform(df[col])
        
        multiclass_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','tenure_group']
    df = pd.get_dummies(df, columns=multiclass_cols, drop_first=True)
    X=df.drop("Churn", axis=1)
    y=df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler=StandardScaler()
    num_cols=['tenure', 'MonthlyCharges', 'TotalCharges','charge_ratio']
    X_train[num_cols]=scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]=scaler.transform(X_test[num_cols])
    print(f"Encoding and scaling completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test 

def save_processed(X_train,X_test,y_train,y_test):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    X_train.to_csv(f"{PROCESSED_DATA_PATH}_X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_PATH}_X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_PATH}_y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_PATH}_y_test.csv", index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    
if __name__ == "__main__":
    df=load_and_clean(RAW_DATA_PATH)
    df=engineer_features(df)
    X_train, X_test, y_train, y_test = encode_and_scale(df)
    save_processed(X_train, X_test, y_train, y_test)