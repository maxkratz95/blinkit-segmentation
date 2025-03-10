from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
from app.models.ml_model import CustomerSegmentationModel
from app.services.data_processor import DataProcessor
import os
from werkzeug.utils import secure_filename
from app.config import Config

web_bp = Blueprint('web', __name__)
model = CustomerSegmentationModel(Config./Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/ml_model.py)

@web_bp.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            # Read and process the CSV file
            df = pd.read_csv(file)
            
            try:
                # Validate and process data
                DataProcessor.validate_data(df)
                processed_df = DataProcessor.process_data(df)
                
                # Get predictions
                segments = model.predict(processed_df)
                processed_df['segment'] = segments
                
                # Store the processed dataframe in session
                # (In production, you might want to use a database instead)
                session['processed_data'] = processed_df.to_json()
                
                return redirect(url_for('web.dashboard'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
                
    return render_template('upload.html')

@web_bp.route('/dashboard')
def dashboard():
    if 'processed_data' not in session:
        return redirect(url_for('web.upload'))
        
    df = pd.read_json(session['processed_data'])
    
    # Prepare dashboard data
    segment_distribution = df['segment'].value_counts().to_dict()
    avg_order_by_segment = df.groupby('segment')['avg_order_value'].mean().to_dict()
    
    return render_template('dashboard.html',
                         segment_distribution=segment_distribution,
                         avg_order_by_segment=avg_order_by_segment,
                         data=df.to_dict('records')) 