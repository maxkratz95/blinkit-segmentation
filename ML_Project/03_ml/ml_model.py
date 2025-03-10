import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

model_path = '/Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/customer_segmentation_model.pkl'
scaler_path = '/Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/scaler.pkl'

class CustomerSegmentationModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = [
            'total_orders', 'avg_order_value', 'tenure_days',
            'recency_days', 'total_order_value'
        ]
        
    def preprocess_data(self, df):
        # Create copy of input features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
        
    def predict(self, df):
        X_scaled = self.preprocess_data(df)
        predictions = self.model.predict(X_scaled)
        
        # Map numerical predictions to segment labels
        segment_mapping = {
            0: 'Inactive',
            1: 'New',
            2: 'Regular',
            3: 'Premium'
        }
        
        return [segment_mapping[pred] for pred in predictions] 