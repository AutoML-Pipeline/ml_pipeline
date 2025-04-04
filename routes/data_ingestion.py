from flask import Blueprint, request, jsonify, current_app
import os
import logging
from controllers.data_controller import DataController

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
data_ingestion_bp = Blueprint('data_ingestion', __name__, url_prefix='/data_ingestion')

# We'll initialize the controller inside each route function
# to ensure we have access to the application context

@data_ingestion_bp.route('', methods=['POST'])
def ingest_data():
    """
    Upload and store a dataset in MinIO.
    
    Expected form data:
    - file: The file to be uploaded
    - dataset_name: A name for the dataset
    - description: (Optional) A description of the dataset
    """
    try:
        # Initialize controller with application context
        data_controller = DataController()
        
        # Check if request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if dataset name is provided
        dataset_name = request.form.get('dataset_name')
        if not dataset_name:
            return jsonify({"error": "Dataset name is required"}), 400
        
        # Get description if provided
        description = request.form.get('description', '')
        
        # Check file extension
        allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'csv', 'parquet', 'json'})
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"File format not supported. Allowed formats: {', '.join(allowed_extensions)}"}), 400
        
        # Process the upload
        result = data_controller.upload_dataset(file, dataset_name, description, file_ext)
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        return jsonify({"error": "Failed to process data ingestion", "details": str(e)}), 500

@data_ingestion_bp.route('', methods=['GET'])
def list_datasets():
    """List all available datasets"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        datasets = data_controller.list_datasets()
        return jsonify({"datasets": datasets}), 200
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return jsonify({"error": "Failed to list datasets", "details": str(e)}), 500

@data_ingestion_bp.route('/<dataset_id>', methods=['GET'])
def get_dataset_info(dataset_id):
    """Get information about a specific dataset"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        dataset_info = data_controller.get_dataset_info(dataset_id)
        if dataset_info:
            return jsonify(dataset_info), 200
        else:
            return jsonify({"error": "Dataset not found"}), 404
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return jsonify({"error": "Failed to get dataset info", "details": str(e)}), 500

@data_ingestion_bp.route('/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a specific dataset"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        success = data_controller.delete_dataset(dataset_id)
        if success:
            return jsonify({"message": "Dataset deleted successfully"}), 200
        else:
            return jsonify({"error": "Dataset not found or could not be deleted"}), 404
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        return jsonify({"error": "Failed to delete dataset", "details": str(e)}), 500
