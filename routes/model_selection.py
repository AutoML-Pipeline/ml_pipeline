from flask import Blueprint, request, jsonify, current_app
import logging
from controllers.model_controller import ModelController

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
model_selection_bp = Blueprint('model_selection', __name__, url_prefix='/model_selection')

# We'll initialize the controller inside each route function
# to ensure we have access to the application context

@model_selection_bp.route('/models', methods=['GET'])
def list_available_models():
    """List all available ML models for training"""
    try:
        # Initialize controller with application context
        model_controller = ModelController()
        
        available_models = model_controller.get_available_models()
        return jsonify({"available_models": available_models}), 200
    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        return jsonify({"error": "Failed to list available models", "details": str(e)}), 500

@model_selection_bp.route('/recommend', methods=['POST'])
def recommend_model():
    """
    Recommend a model based on dataset characteristics and problem type
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "target_column": "target",
        "problem_type": "classification" | "regression",
        "evaluation_metric": "accuracy" | "f1" | "rmse" | "mae" | "r2"
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        target_column = req_data.get('target_column')
        problem_type = req_data.get('problem_type')
        evaluation_metric = req_data.get('evaluation_metric')
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
            
        if not problem_type:
            return jsonify({"error": "Problem type is required"}), 400
            
        if problem_type not in ['classification', 'regression']:
            return jsonify({"error": "Problem type must be either 'classification' or 'regression'"}), 400
        
        # Initialize controller with application context
        model_controller = ModelController()
        
        # Get model recommendation
        recommendation = model_controller.recommend_model(
            dataset_id,
            target_column,
            problem_type,
            evaluation_metric
        )
        
        return jsonify(recommendation), 200
        
    except Exception as e:
        logger.error(f"Error in model recommendation: {str(e)}")
        return jsonify({"error": "Failed to recommend model", "details": str(e)}), 500

@model_selection_bp.route('/hyperparameters', methods=['GET'])
def get_model_hyperparameters():
    """
    Get available hyperparameters for a specific model
    
    Query parameters:
    - model_name: Name of the model (e.g., random_forest_classifier)
    """
    try:
        model_name = request.args.get('model_name')
        
        if not model_name:
            return jsonify({"error": "Model name is required"}), 400
            
        # Initialize controller with application context
        model_controller = ModelController()
        
        hyperparameters = model_controller.get_model_hyperparameters(model_name)
        
        if hyperparameters:
            return jsonify(hyperparameters), 200
        else:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting model hyperparameters: {str(e)}")
        return jsonify({"error": "Failed to get model hyperparameters", "details": str(e)}), 500

@model_selection_bp.route('/compare', methods=['POST'])
def compare_models():
    """
    Compare multiple models on a dataset
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "target_column": "target",
        "problem_type": "classification" | "regression",
        "models": [
            {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 100}},
            {"name": "xgboost_classifier", "hyperparameters": {"n_estimators": 100}}
        ],
        "evaluation_metrics": ["accuracy", "f1", "precision", "recall"],
        "cv_folds": 5
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        target_column = req_data.get('target_column')
        problem_type = req_data.get('problem_type')
        models = req_data.get('models', [])
        evaluation_metrics = req_data.get('evaluation_metrics', [])
        cv_folds = req_data.get('cv_folds', 5)
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
            
        if not problem_type:
            return jsonify({"error": "Problem type is required"}), 400
            
        if not models:
            return jsonify({"error": "At least one model is required"}), 400
            
        if not evaluation_metrics:
            return jsonify({"error": "At least one evaluation metric is required"}), 400
        
        # Initialize controller with application context
        model_controller = ModelController()
        
        # Compare models
        comparison_results = model_controller.compare_models(
            dataset_id,
            target_column,
            problem_type,
            models,
            evaluation_metrics,
            cv_folds
        )
        
        return jsonify(comparison_results), 200
        
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return jsonify({"error": "Failed to compare models", "details": str(e)}), 500
