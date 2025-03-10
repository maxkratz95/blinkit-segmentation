from flask import Flask
from .routes import main_bp

def create_app():
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')
    
    # Configure your app
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Register blueprints
    app.register_blueprint(main_bp)
    
    return app