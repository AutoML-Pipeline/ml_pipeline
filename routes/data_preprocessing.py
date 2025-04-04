from flask import Blueprint, request, jsonify, current_app
import logging
from controllers.data_controller import DataController

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
data_preprocessing_bp = Blueprint('data_preprocessing', __name__, url_prefix='/data_preprocessing')

# We'll initialize the controller inside each route function
# to ensure we have access to the application context

@data_preprocessing_bp.route('', methods=['POST'])
def preprocess_data():
    """
    Perform data preprocessing operations on a dataset.
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "operations": [
            {"type": "remove_nulls", "columns": ["col1", "col2"]},
            {"type": "fillna", "columns": ["col3"], "value": 0},
            {"type": "normalize", "columns": ["col4", "col5"], "method": "minmax"},
            {"type": "outlier_removal", "columns": ["col6"], "method": "iqr", "threshold": 1.5}
        ],
        "output_name": "preprocessed_dataset_name"
    }
    """
    try:
        # Parse request data
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        operations = req_data.get('operations', [])
        output_name = req_data.get('output_name')
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not operations:
            return jsonify({"error": "At least one preprocessing operation is required"}), 400
            
        if not output_name:
            return jsonify({"error": "Output dataset name is required"}), 400
        
        # Initialize controller with application context
        data_controller = DataController()
        
        # Execute preprocessing
        result = data_controller.preprocess_dataset(dataset_id, operations, output_name)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return jsonify({"error": "Failed to process data preprocessing", "details": str(e)}), 500

@data_preprocessing_bp.route('/operations', methods=['GET'])
def list_preprocessing_operations():
    """List all available preprocessing operations"""
    operations = {
        "remove_nulls": "Remove rows with null values in specified columns",
        "fillna": "Fill null values with specified value",
        "normalize": "Normalize values using specified method (minmax, zscore)",
        "outlier_removal": "Remove outliers using specified method (iqr, zscore)",
        "categorical_encoding": "Encode categorical variables (one-hot, label)",
        "datetime_features": "Extract features from datetime columns",
        "binning": "Bin continuous variables into discrete categories"
    }
    
    return jsonify({"available_operations": operations}), 200

@data_preprocessing_bp.route('/dataset/<dataset_id>/preview', methods=['GET'])
def preview_dataset(dataset_id):
    """Get a preview of the dataset (first few rows)"""
    try:
        # Get optional limit parameter
        limit = request.args.get('limit', default=10, type=int)
        
        # Initialize controller with application context
        data_controller = DataController()
        
        # Get preview data
        preview_data = data_controller.get_dataset_preview(dataset_id, limit)
        
        if preview_data:
            return jsonify(preview_data), 200
        else:
            return jsonify({"error": "Dataset not found or cannot be previewed"}), 404
            
    except Exception as e:
        logger.error(f"Error getting dataset preview: {str(e)}")
        return jsonify({"error": "Failed to get dataset preview", "details": str(e)}), 500

@data_preprocessing_bp.route('/dataset/<dataset_id>/schema', methods=['GET'])
def get_dataset_schema(dataset_id):
    """Get the schema of the dataset (column names and types)"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        
        schema = data_controller.get_dataset_schema(dataset_id)
        
        if schema:
            return jsonify(schema), 200
        else:
            return jsonify({"error": "Dataset not found or schema cannot be retrieved"}), 404
            
    except Exception as e:
        logger.error(f"Error getting dataset schema: {str(e)}")
        return jsonify({"error": "Failed to get dataset schema", "details": str(e)}), 500

@data_preprocessing_bp.route('/job/<job_id>', methods=['GET'])
def get_preprocessing_job_status(job_id):
    """Get the status of a preprocessing job"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        
        job_status = data_controller.get_preprocessing_job_status(job_id)
        
        if job_status:
            return jsonify(job_status), 200
        else:
            return jsonify({"error": "Job not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting preprocessing job status: {str(e)}")
        return jsonify({"error": "Failed to get preprocessing job status", "details": str(e)}), 500
