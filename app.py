import os
import logging
from flask import Flask, render_template, jsonify, flash, session
import config
from routes.data_ingestion import data_ingestion_bp
from routes.data_preprocessing import data_preprocessing_bp
from routes.feature_engineering import feature_engineering_bp
from routes.model_selection import model_selection_bp
# Import the following route blueprints as they are implemented
# from routes.model_training_evaluation import model_training_evaluation_bp
# from routes.visualization import visualization_bp
# from routes.deployment import deployment_bp

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-key")

# Load configuration from config.py
app.config.from_object(config)

# Register blueprints with error handling
try:
    app.register_blueprint(data_ingestion_bp)
    app.register_blueprint(data_preprocessing_bp)
    app.register_blueprint(feature_engineering_bp)
    app.register_blueprint(model_selection_bp)
    # Register the following blueprints as they are implemented
    # app.register_blueprint(model_training_evaluation_bp)
    # app.register_blueprint(visualization_bp)
    # app.register_blueprint(deployment_bp)
    logger.info("All blueprints registered successfully")
except Exception as e:
    logger.error(f"Error registering blueprints: {str(e)}")
    # We'll continue with the application so the basic UI is available

# Set up services and check their status
app.config['SERVICES_STATUS'] = {
    'minio': False,
    'spark': False,
    'database': False
}

# Use mock services by default for development
app.config['USE_MOCK_SERVICES'] = True

# Configure MinIO settings
# Using provided credentials
app.config['MINIO_ENDPOINT'] = '127.0.0.1:9090'  # Removing the /buckets part
app.config['MINIO_ACCESS_KEY'] = ''  # Public bucket, no access key needed
app.config['MINIO_SECRET_KEY'] = ''  # Public bucket, no secret key needed
app.config['MINIO_SECURE'] = False  # Using http, not https
app.config['MINIO_BUCKET_NAME'] = 'mlpipeline'

# Try to initialize MinIO connection
try:
    from services.minio_service import MinioService
    minio_service = MinioService(
        endpoint=app.config['MINIO_ENDPOINT'],
        access_key=app.config['MINIO_ACCESS_KEY'],
        secret_key=app.config['MINIO_SECRET_KEY'],
        secure=app.config['MINIO_SECURE']
    )
    app.config['SERVICES_STATUS']['minio'] = True
    logger.info("MinIO service initialized successfully")
except Exception as e:
    logger.warning(f"MinIO service initialization failed: {str(e)}. Some features may be unavailable.")
    # Load the mock service as a fallback
    try:
        from services.mock_minio_service import MockMinioService
        logger.info("Using MockMinioService as a fallback")
    except Exception as mock_error:
        logger.error(f"Failed to load MockMinioService: {str(mock_error)}")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# API documentation
@app.route('/api/docs')
def api_docs():
    return render_template('api_docs.html')

# Services status endpoint
@app.route('/api/services/status')
def services_status():
    return jsonify(app.config['SERVICES_STATUS'])

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return {"error": "Internal server error", "message": str(e)}, 500
