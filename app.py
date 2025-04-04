import os
import logging
from flask import Flask
from routes.data_ingestion import data_ingestion_bp
from routes.data_preprocessing import data_preprocessing_bp
from routes.feature_engineering import feature_engineering_bp
from routes.model_selection import model_selection_bp
from routes.model_training_evaluation import model_training_evaluation_bp
from routes.visualization import visualization_bp
from routes.deployment import deployment_bp

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Register blueprints
app.register_blueprint(data_ingestion_bp, url_prefix='/api')
app.register_blueprint(data_preprocessing_bp, url_prefix='/api')
app.register_blueprint(feature_engineering_bp, url_prefix='/api')
app.register_blueprint(model_selection_bp, url_prefix='/api')
app.register_blueprint(model_training_evaluation_bp, url_prefix='/api')
app.register_blueprint(visualization_bp, url_prefix='/api')
app.register_blueprint(deployment_bp, url_prefix='/api')

# Root route
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return {"error": "Not found"}, 404

@app.errorhandler(500)
def internal_error(error):
    return {"error": "Internal server error"}, 500
