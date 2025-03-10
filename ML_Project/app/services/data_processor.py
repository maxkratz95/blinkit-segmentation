import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def validate_data(df):
        required_columns = [
            'customer_id', 'total_orders', 'avg_order_value',
            'tenure_days', 'recency_days', 'total_order_value'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    @staticmethod
    def process_data(df):
        # Create copy of dataframe
        processed_df = df.copy()
        
        # Handle missing values
        numeric_columns = ['total_orders', 'avg_order_value', 'tenure_days', 
                         'recency_days', 'total_order_value']
        processed_df[numeric_columns] = processed_df[numeric_columns].fillna(
            processed_df[numeric_columns].mean()
        )
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['customer_id'])
        
        return processed_df 