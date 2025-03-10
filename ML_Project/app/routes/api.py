from flask import Blueprint, jsonify, request
from app.models.ml_model import CustomerSegmentationModel
from app.services.data_processor import DataProcessor
import pandas as pd
from app.config import Config

MODEL_PATH = '/Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/customer_segmentation_model.pkl'

api_bp = Blueprint('api', __name__)
model = CustomerSegmentationModel(Config.MODEL_PATH)

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Validate and process data
        DataProcessor.validate_data(df)
        processed_df = DataProcessor.process_data(df)
        
        # Get predictions
        segments = model.predict(processed_df)
        
        # Prepare response
        response = {
            'success': True,
            'predictions': [
                {
                    'customer_id': row['customer_id'],
                    'segment': segment
                }
                for row, segment in zip(processed_df.to_dict('records'), segments)
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400 