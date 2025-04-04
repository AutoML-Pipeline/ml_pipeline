from flask import Blueprint, request, jsonify
from controllers.data_ingestion_controller import upload_dataset, list_datasets, get_dataset_info

data_ingestion_bp = Blueprint('data_ingestion', __name__)

@data_ingestion_bp.route('/data_ingestion/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to upload a dataset file (CSV, Parquet, JSON)
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The dataset file to upload
      - name: dataset_name
        in: formData
        type: string
        required: true
        description: Name for the dataset
      - name: description
        in: formData
        type: string
        required: false
        description: Optional description of the dataset
    responses:
        200:
            description: Dataset uploaded successfully
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    dataset_name = request.form.get('dataset_name')
    description = request.form.get('description', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'}), 400
    
    try:
        result = upload_dataset(file, dataset_name, description)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_ingestion_bp.route('/data_ingestion/datasets', methods=['GET'])
def get_datasets():
    """
    Endpoint to list all available datasets
    ---
    responses:
        200:
            description: List of available datasets
    """
    try:
        datasets = list_datasets()
        return jsonify({'datasets': datasets}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_ingestion_bp.route('/data_ingestion/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """
    Endpoint to get information about a specific dataset
    ---
    parameters:
      - name: dataset_id
        in: path
        type: string
        required: true
        description: ID of the dataset
    responses:
        200:
            description: Dataset information
    """
    try:
        dataset = get_dataset_info(dataset_id)
        if dataset:
            return jsonify(dataset), 200
        else:
            return jsonify({'error': 'Dataset not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
