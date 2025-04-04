from flask import Blueprint, request, jsonify
import logging
from controllers.model_controller import ModelController

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
deployment_bp = Blueprint('deployment', __name__, url_prefix='/deployment')

# Initialize controller
model_controller = ModelController()

@deployment_bp.route('/deploy', methods=['POST'])
def deploy_model():
    """
    Deploy a trained ML model for predictions
    
    Expected JSON body:
    {
        "model_id": "12345",
        "deployment_name": "my-model-endpoint",
        "description": "Deployed model for product recommendations",
        "version": "1.0"
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        model_id = req_data.get('model_id')
        deployment_name = req_data.get('deployment_name')
        description = req_data.get('description', '')
        version = req_data.get('version', '1.0')
        
        # Validate required fields
        if not model_id:
            return jsonify({"error": "Model ID is required"}), 400
        
        if not deployment_name:
            return jsonify({"error": "Deployment name is required"}), 400
        
        # Deploy the model
        result = model_controller.deploy_model(
            model_id,
            deployment_name,
            description,
            version
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        return jsonify({"error": "Failed to deploy model", "details": str(e)}), 500

@deployment_bp.route('/list', methods=['GET'])
def list_deployments():
    """List all deployed models"""
    try:
        deployments = model_controller.list_deployments()
        return jsonify({"deployments": deployments}), 200
    except Exception as e:
        logger.error(f"Error listing deployments: {str(e)}")
        return jsonify({"error": "Failed to list deployments", "details": str(e)}), 500

@deployment_bp.route('/<deployment_id>', methods=['GET'])
def get_deployment(deployment_id):
    """Get details of a specific deployment"""
    try:
        deployment = model_controller.get_deployment(deployment_id)
        
        if deployment:
            return jsonify(deployment), 200
        else:
            return jsonify({"error": "Deployment not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting deployment: {str(e)}")
        return jsonify({"error": "Failed to get deployment", "details": str(e)}), 500

@deployment_bp.route('/<deployment_id>', methods=['DELETE'])
def undeploy_model(deployment_id):
    """Undeploy a model"""
    try:
        success = model_controller.undeploy_model(deployment_id)
        
        if success:
            return jsonify({"message": "Model undeployed successfully"}), 200
        else:
            return jsonify({"error": "Deployment not found or could not be undeployed"}), 404
            
    except Exception as e:
        logger.error(f"Error undeploying model: {str(e)}")
        return jsonify({"error": "Failed to undeploy model", "details": str(e)}), 500

@deployment_bp.route('/<deployment_id>/predict', methods=['POST'])
def predict(deployment_id):
    """
    Make predictions using a deployed model
    
    Expected JSON body:
    {
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...}
        ]
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = req_data.get('data', [])
        
        if not data:
            return jsonify({"error": "No data provided for prediction"}), 400
        
        # Make predictions
        predictions = model_controller.predict(deployment_id, data)
        
        return jsonify({"predictions": predictions}), 200
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return jsonify({"error": "Failed to make predictions", "details": str(e)}), 500

@deployment_bp.route('/<deployment_id>/batch_predict', methods=['POST'])
def batch_predict(deployment_id):
    """
    Make batch predictions using a deployed model and a dataset
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "output_dataset_name": "predictions_output"
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        output_name = req_data.get('output_dataset_name')
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not output_name:
            return jsonify({"error": "Output dataset name is required"}), 400
        
        # Make batch predictions
        result = model_controller.batch_predict(
            deployment_id,
            dataset_id,
            output_name
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        return jsonify({"error": "Failed to make batch predictions", "details": str(e)}), 500

@deployment_bp.route('/<deployment_id>/metadata', methods=['GET'])
def get_deployment_metadata(deployment_id):
    """Get metadata for a deployed model"""
    try:
        metadata = model_controller.get_deployment_metadata(deployment_id)
        
        if metadata:
            return jsonify(metadata), 200
        else:
            return jsonify({"error": "Deployment not found or metadata not available"}), 404
            
    except Exception as e:
        logger.error(f"Error getting deployment metadata: {str(e)}")
        return jsonify({"error": "Failed to get deployment metadata", "details": str(e)}), 500
