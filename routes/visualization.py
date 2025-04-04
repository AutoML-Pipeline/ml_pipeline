from flask import Blueprint, request, jsonify, send_file
import logging
from controllers.model_controller import ModelController
from services.visualization_service import VisualizationService

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
visualization_bp = Blueprint('visualization', __name__, url_prefix='/visualization')

# Initialize services
model_controller = ModelController()
visualization_service = VisualizationService()

@visualization_bp.route('/data', methods=['POST'])
def visualize_data():
    """
    Generate data visualizations
    
    Expected JSON body:
    {
        "dataset_id": "12345",
        "visualization_type": "histogram" | "scatter" | "bar" | "box" | "heatmap" | "pie" | "line",
        "columns": ["column1", "column2"],
        "title": "My Visualization",
        "options": {
            "x_label": "X Axis",
            "y_label": "Y Axis",
            "color": "blue",
            "bins": 20,  # for histograms
            "figsize": [10, 6]
        }
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        dataset_id = req_data.get('dataset_id')
        viz_type = req_data.get('visualization_type')
        columns = req_data.get('columns', [])
        title = req_data.get('title', 'Data Visualization')
        options = req_data.get('options', {})
        
        # Validate required fields
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required"}), 400
        
        if not viz_type:
            return jsonify({"error": "Visualization type is required"}), 400
            
        if not columns:
            return jsonify({"error": "At least one column is required"}), 400
        
        # Generate visualization
        viz_result = visualization_service.create_data_visualization(
            dataset_id,
            viz_type,
            columns,
            title,
            options
        )
        
        return jsonify(viz_result), 200
        
    except Exception as e:
        logger.error(f"Error generating data visualization: {str(e)}")
        return jsonify({"error": "Failed to generate visualization", "details": str(e)}), 500

@visualization_bp.route('/model/<model_id>', methods=['POST'])
def visualize_model(model_id):
    """
    Generate model visualizations
    
    Expected JSON body:
    {
        "visualization_type": "confusion_matrix" | "roc_curve" | "precision_recall" | "feature_importance" | "learning_curve",
        "title": "My Model Visualization",
        "options": {
            "figsize": [10, 6],
            "cmap": "Blues"  # for confusion matrix
        }
    }
    """
    try:
        req_data = request.get_json()
        
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        viz_type = req_data.get('visualization_type')
        title = req_data.get('title', 'Model Visualization')
        options = req_data.get('options', {})
        
        # Validate required fields
        if not viz_type:
            return jsonify({"error": "Visualization type is required"}), 400
        
        # Generate visualization
        viz_result = visualization_service.create_model_visualization(
            model_id,
            viz_type,
            title,
            options
        )
        
        return jsonify(viz_result), 200
        
    except Exception as e:
        logger.error(f"Error generating model visualization: {str(e)}")
        return jsonify({"error": "Failed to generate visualization", "details": str(e)}), 500

@visualization_bp.route('/image/<visualization_id>', methods=['GET'])
def get_visualization_image(visualization_id):
    """Get visualization image by ID"""
    try:
        image_path = visualization_service.get_visualization_image_path(visualization_id)
        
        if image_path:
            return send_file(image_path, mimetype='image/png')
        else:
            return jsonify({"error": "Visualization not found"}), 404
            
    except Exception as e:
        logger.error(f"Error retrieving visualization image: {str(e)}")
        return jsonify({"error": "Failed to retrieve visualization", "details": str(e)}), 500

@visualization_bp.route('/list', methods=['GET'])
def list_visualizations():
    """List all available visualizations"""
    try:
        # Get optional filters
        dataset_id = request.args.get('dataset_id')
        model_id = request.args.get('model_id')
        viz_type = request.args.get('type')
        
        visualizations = visualization_service.list_visualizations(dataset_id, model_id, viz_type)
        return jsonify({"visualizations": visualizations}), 200
    except Exception as e:
        logger.error(f"Error listing visualizations: {str(e)}")
        return jsonify({"error": "Failed to list visualizations", "details": str(e)}), 500

@visualization_bp.route('/<visualization_id>', methods=['DELETE'])
def delete_visualization(visualization_id):
    """Delete a visualization"""
    try:
        success = visualization_service.delete_visualization(visualization_id)
        
        if success:
            return jsonify({"message": "Visualization deleted successfully"}), 200
        else:
            return jsonify({"error": "Visualization not found or could not be deleted"}), 404
            
    except Exception as e:
        logger.error(f"Error deleting visualization: {str(e)}")
        return jsonify({"error": "Failed to delete visualization", "details": str(e)}), 500

@visualization_bp.route('/types', methods=['GET'])
def get_visualization_types():
    """Get available visualization types"""
    try:
        data_viz_types = {
            "histogram": "Distribution of a numerical variable",
            "scatter": "Relationship between two numerical variables",
            "bar": "Comparison of categorical variables",
            "box": "Distribution and outliers of numerical variables",
            "heatmap": "Correlation between variables",
            "pie": "Proportion of categories",
            "line": "Trends over time or sequence"
        }
        
        model_viz_types = {
            "confusion_matrix": "Visualization of true vs predicted values",
            "roc_curve": "Receiver Operating Characteristic curve",
            "precision_recall": "Precision-Recall curve",
            "feature_importance": "Importance of each feature in the model",
            "learning_curve": "Model performance vs training set size",
            "residuals": "Residual plot for regression models",
            "decision_boundary": "Decision boundary for classification models"
        }
        
        return jsonify({
            "data_visualizations": data_viz_types,
            "model_visualizations": model_viz_types
        }), 200
    except Exception as e:
        logger.error(f"Error getting visualization types: {str(e)}")
        return jsonify({"error": "Failed to get visualization types", "details": str(e)}), 500
