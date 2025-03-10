from flask import Blueprint, jsonify, render_template, request, flash, redirect, url_for
import pandas as pd
import os
import pickle
from werkzeug.utils import secure_filename
import traceback  # Add this for detailed error tracking

main_bp = Blueprint('main', __name__)

# Debug: Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Use absolute paths for all files
MODEL_PATH = '/Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/customer_segmentation_model.pkl'
SCALER_PATH = '/Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/scaler.pkl'
DATASET_PATH = '/Users/a12345/Desktop/DATA_PT/ML_Project/03_ml/preprocessed_dataset_v2.csv'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

print(f"Looking for files at:")
print(f"Model: {MODEL_PATH}")
print(f"Scaler: {SCALER_PATH}")
print(f"Dataset: {DATASET_PATH}")
print(f"Upload folder: {UPLOAD_FOLDER}")

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Debug: Check if files exist
print(f"File existence check:")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Scaler exists: {os.path.exists(SCALER_PATH)}")
print(f"Dataset exists: {os.path.exists(DATASET_PATH)}")

# Load both model and scaler
try:
    print("Attempting to load model...")
    # Try different pickle protocols
    import _pickle as cPickle
    with open(MODEL_PATH, 'rb') as file:
        model = cPickle.load(file)
    print(f"Model type: {type(model)}")
    
    print("Attempting to load scaler...")
    with open(SCALER_PATH, 'rb') as file:
        scaler = cPickle.load(file)
    print(f"Scaler type: {type(scaler)}")
    
    print("Both model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    print(f"Full traceback: {traceback.format_exc()}")
    try:
        # Try alternative loading method
        import joblib
        print("Attempting to load with joblib...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Loaded successfully with joblib!")
    except Exception as e2:
        print(f"Joblib loading also failed: {str(e2)}")
        model = None
        scaler = None

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this mapping at the top of your file after the imports
CLUSTER_NAMES = {
    0: "High-Spending Frequent Buyers",
    1: "VIP Repeat Customers",
    2: "High-Spending Frequent Buyers",
    3: "High-Spending Frequent Buyers"
}

@main_bp.route('/')
@main_bp.route('/dashboard')
def home():
    try:
        df = pd.read_csv(DATASET_PATH)
        
        # Define cluster names
        cluster_names = {
            0: "VIP Repeat Customers",
            1: "High-Spending Frequent Buyers",
            2: "One-Time High Spenders",
            3: "Low-Spending & Inactive Users"
        }
        
        # Map numeric segments to names
        df['segment_name'] = df['customer_segment'].map(cluster_names)
        
        return render_template('dashboard.html',
                             cluster_names=cluster_names,
                             data=df.to_dict('records'),
                             is_uploaded_data=False)
    except Exception as e:
        print(f"Error in home route: {str(e)}")
        return render_template('dashboard.html',
                             cluster_names={},
                             data=[],
                             error=str(e),
                             is_uploaded_data=False)

@main_bp.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            "success": True,
            "prediction": prediction.tolist(),
            "segment": prediction[0]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400 

@main_bp.route('/upload_file', methods=['POST'])
def upload_file():
    print("Upload file route hit")
    
    if 'file' not in request.files:
        print("No file in request")
        flash('No file part')
        return redirect(url_for('main.home'))
    
    file = request.files['file']
    print(f"Received file: {file.filename}")
    
    if file.filename == '':
        print("Empty filename")
        flash('No selected file')
        return redirect(url_for('main.home'))
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Read and process file
            print("Reading uploaded file")
            df = pd.read_csv(filepath)
            print(f"Uploaded file columns: {df.columns.tolist()}")
            
            if model is not None and scaler is not None:
                required_features = ['total_spent', 'avg_order_value', 'order_count', 'avg_delivery_time']
                
                # Check features
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    print(f"Missing features: {missing_features}")
                    flash(f'Missing columns: {", ".join(missing_features)}')
                    return redirect(url_for('main.home'))
                
                print("Scaling features...")
                scaled_features = scaler.transform(df[required_features])
                
                print("Making predictions...")
                predictions = model.predict(scaled_features)
                df['predicted_segment'] = predictions
                
                print(f"Predictions made: {predictions[:5]}")  # Show first 5 predictions
                
                # Define cluster names
                cluster_names = {
                    0: "VIP Repeat Customers",
                    1: "High-Spending Frequent Buyers",
                    2: "One-Time High Spenders",
                    3: "Low-Spending & Inactive Users"
                }
                
                # Define stats for each cluster
                stats = {
                    0: {'avg_spend': '~$23,794', 'avg_orders': '~15.8', 'avg_value': '~$1,520'},
                    1: {'avg_spend': '~$6,750', 'avg_orders': '~9.5', 'avg_value': '~$1,050'},
                    2: {'avg_spend': '~$3,500', 'avg_orders': '~1.5', 'avg_value': '~$3,000'},
                    3: {'avg_spend': '~$1,250', 'avg_orders': '~0.75', 'avg_value': '~$500'}
                }
                
                # Map numeric predictions to cluster names
                df['predicted_segment_name'] = df['predicted_segment'].map(cluster_names)
                
                # Calculate distributions using named segments
                segment_distribution = df['predicted_segment_name'].value_counts().to_dict()
                avg_order_by_segment = df.groupby('predicted_segment_name')['avg_order_value'].mean().to_dict()
                
                # Save processed file
                output_filepath = os.path.join(UPLOAD_FOLDER, 'processed_' + filename)
                df.to_csv(output_filepath, index=False)
                print(f"Saved processed file to: {output_filepath}")
                
                return render_template('dashboard.html',
                                    cluster_names=cluster_names,
                                    stats=stats,
                                    segment_distribution=segment_distribution,
                                    avg_order_by_segment=avg_order_by_segment,
                                    data=df.to_dict('records'),
                                    is_uploaded_data=True)
            else:
                print("Model or scaler not loaded")
                flash('Model or scaler not loaded, file was uploaded but not processed')
                
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            flash(f'Error processing file: {str(e)}')
            
        return redirect(url_for('main.home'))
    
    print("Invalid file type")
    flash('Invalid file type')
    return redirect(url_for('main.home')) 