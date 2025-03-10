import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uploads')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models', 'customer_segmentation_model.pkl')
    ALLOWED_EXTENSIONS = {'csv'} 