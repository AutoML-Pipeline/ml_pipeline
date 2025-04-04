from flask import Blueprint, request, jsonify
import logging
from controllers.model_controller import ModelController

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
model_training_evaluation_bp = Blueprint('model_training_evaluation', __name__, url_prefix='/model_training_evaluation')

# Initialize controller
model_controller = ModelController()

@model_training_evaluation_bp.route('/train', methods=['POST'])
def train_model():
    """
    Train a machine learning model
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2", ...], (optional, if not provided all columns except target will be used)
        "model_name": "random_forest_classifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "test_size": 0.2,
        "random_state": 42,
        "model_description": "Random Forest model for predicting X"
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        target_column = req_data.get('target_column')
        feature_columns = req_data.get('feature_columns')
        model_name = req_data.get('model_name')
        hyperparameters = req_data.get('hyperparameters', {})
        test_size = req_data.get('test_size', 0.2)
        random_state = req_data.get('random_state', 42)
        model_description = req_data.get('model_description', '')
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
            
        if not model_name:
            return jsonify({"error": "Model name is required"}), 400
        
        # Train the model
        result = model_controller.train_model(
            dataset_id,
            target_column,
            feature_columns,
            model_name,
            hyperparameters,
            test_size,
            random_state,
            model_description
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({"error": "Failed to train model", "details": str(e)}), 500

@model_training_evaluation_bp.route('/evaluate/<model_id>', methods=['GET'])
def get_model_evaluation(model_id):
    """Get evaluation metrics for a trained model"""
    try:
        evaluation_results = model_controller.get_model_evaluation(model_id)
        
        if evaluation_results:
            return jsonify(evaluation_results), 200
        else:
            return jsonify({"error": "Model not found or evaluation results not available"}), 404
            
    except Exception as e:
        logger.error(f"Error getting model evaluation: {str(e)}")
        return jsonify({"error": "Failed to get model evaluation", "details": str(e)}), 500

@model_training_evaluation_bp.route('/models', methods=['GET'])
def list_trained_models():
    """List all trained models"""
    try:
        trained_models = model_controller.list_trained_models()
        return jsonify({"trained_models": trained_models}), 200
    except Exception as e:
        logger.error(f"Error listing trained models: {str(e)}")
        return jsonify({"error": "Failed to list trained models", "details": str(e)}), 500

@model_training_evaluation_bp.route('/models/<model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get details of a specific trained model"""
    try:
        model_details = model_controller.get_model_details(model_id)
        
        if model_details:
            return jsonify(model_details), 200
        else:
            return jsonify({"error": "Model not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        return jsonify({"error": "Failed to get model details", "details": str(e)}), 500

@model_training_evaluation_bp.route('/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a trained model"""
    try:
        success = model_controller.delete_model(model_id)
        
        if success:
            return jsonify({"message": "Model deleted successfully"}), 200
        else:
            return jsonify({"error": "Model not found or could not be deleted"}), 404
            
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return jsonify({"error": "Failed to delete model", "details": str(e)}), 500

@model_training_evaluation_bp.route('/cross_validate', methods=['POST'])
def cross_validate_model():
    """
    Perform cross-validation for a model
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2", ...], (optional)
        "model_name": "random_forest_classifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "cv_folds": 5,
        "evaluation_metrics": ["accuracy", "f1", "precision", "recall"],
        "random_state": 42
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        target_column = req_data.get('target_column')
        feature_columns = req_data.get('feature_columns')
        model_name = req_data.get('model_name')
        hyperparameters = req_data.get('hyperparameters', {})
        cv_folds = req_data.get('cv_folds', 5)
        evaluation_metrics = req_data.get('evaluation_metrics', [])
        random_state = req_data.get('random_state', 42)
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
            
        if not model_name:
            return jsonify({"error": "Model name is required"}), 400
            
        if not evaluation_metrics:
            return jsonify({"error": "At least one evaluation metric is required"}), 400
        
        # Perform cross-validation
        cv_results = model_controller.cross_validate_model(
            dataset_id,
            target_column,
            feature_columns,
            model_name,
            hyperparameters,
            cv_folds,
            evaluation_metrics,
            random_state
        )
        
        return jsonify(cv_results), 200
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return jsonify({"error": "Failed to perform cross-validation", "details": str(e)}), 500

@model_training_evaluation_bp.route('/job/<job_id>', methods=['GET'])
def get_training_job_status(job_id):
    """Get the status of a model training job"""
    try:
        job_status = model_controller.get_training_job_status(job_id)
        
        if job_status:
            return jsonify(job_status), 200
        else:
            return jsonify({"error": "Job not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting training job status: {str(e)}")
        return jsonify({"error": "Failed to get training job status", "details": str(e)}), 500
