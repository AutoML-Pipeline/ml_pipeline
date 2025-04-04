from flask import Blueprint, request, jsonify, current_app
import logging
from controllers.data_controller import DataController

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
feature_engineering_bp = Blueprint('feature_engineering', __name__, url_prefix='/feature_engineering')

# We'll initialize the controller inside each route function
# to ensure we have access to the application context

@feature_engineering_bp.route('', methods=['POST'])
def engineer_features():
    """
    Perform feature engineering operations on a dataset.
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "operations": [
            {"type": "polynomial_features", "columns": ["col1", "col2"], "degree": 2},
            {"type": "binning", "column": "col3", "bins": 5, "labels": ["v_low", "low", "medium", "high", "v_high"]},
            {"type": "text_vectorization", "column": "text_col", "method": "tfidf", "max_features": 1000},
            {"type": "pca", "columns": ["col4", "col5", "col6"], "n_components": 2}
        ],
        "output_name": "featured_dataset_name"
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
            return jsonify({"error": "At least one feature engineering operation is required"}), 400
            
        if not output_name:
            return jsonify({"error": "Output dataset name is required"}), 400
        
        # Initialize controller with application context
        data_controller = DataController()
        
        # Execute feature engineering
        result = data_controller.engineer_features(dataset_id, operations, output_name)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        return jsonify({"error": "Failed to perform feature engineering", "details": str(e)}), 500

@feature_engineering_bp.route('/operations', methods=['GET'])
def list_feature_engineering_operations():
    """List all available feature engineering operations"""
    operations = {
        "polynomial_features": "Create polynomial features from specified columns",
        "binning": "Bin numerical features into categorical bins",
        "text_vectorization": "Convert text to numerical features (tfidf, count, word2vec)",
        "pca": "Perform Principal Component Analysis for dimensionality reduction",
        "feature_selection": "Select features based on importance (chi2, f_value, mutual_info)",
        "interaction_features": "Create interaction features between columns",
        "time_features": "Extract time-based features from datetime columns",
        "aggregation": "Aggregate features by group (mean, sum, min, max, count)",
        "lag_features": "Create lag features for time series data",
        "custom_transformer": "Apply custom transformation function"
    }
    
    return jsonify({"available_operations": operations}), 200

@feature_engineering_bp.route('/dataset/<dataset_id>/features', methods=['GET'])
def get_dataset_features(dataset_id):
    """Get the list of features in the dataset"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        
        features = data_controller.get_dataset_features(dataset_id)
        
        if features:
            return jsonify(features), 200
        else:
            return jsonify({"error": "Dataset not found or features cannot be retrieved"}), 404
            
    except Exception as e:
        logger.error(f"Error getting dataset features: {str(e)}")
        return jsonify({"error": "Failed to get dataset features", "details": str(e)}), 500

@feature_engineering_bp.route('/dataset/<dataset_id>/feature_importance', methods=['POST'])
def get_feature_importance(dataset_id):
    """
    Get feature importance for a dataset
    
    Expected JSON body:
    {
        "target_column": "target",
        "method": "mutual_info" | "chi2" | "f_classif" | "f_regression"
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        target_column = req_data.get('target_column')
        method = req_data.get('method', 'mutual_info')
        
        # Validate required fields
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
        
        # Initialize controller with application context
        data_controller = DataController()
        
        # Get feature importance
        feature_importance = data_controller.calculate_feature_importance(
            dataset_id, 
            target_column, 
            method
        )
        
        if feature_importance:
            return jsonify(feature_importance), 200
        else:
            return jsonify({"error": "Failed to calculate feature importance"}), 400
            
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return jsonify({"error": "Failed to calculate feature importance", "details": str(e)}), 500

@feature_engineering_bp.route('/job/<job_id>', methods=['GET'])
def get_feature_engineering_job_status(job_id):
    """Get the status of a feature engineering job"""
    try:
        # Initialize controller with application context
        data_controller = DataController()
        
        job_status = data_controller.get_feature_engineering_job_status(job_id)
        
        if job_status:
            return jsonify(job_status), 200
        else:
            return jsonify({"error": "Job not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting feature engineering job status: {str(e)}")
        return jsonify({"error": "Failed to get feature engineering job status", "details": str(e)}), 500
