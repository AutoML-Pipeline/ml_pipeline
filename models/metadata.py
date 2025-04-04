from datetime import datetime
import json

class DatasetMetadata:
    """Class to represent dataset metadata"""
    
    def __init__(self, id, name, description, original_filename, storage_filename, 
                 file_type, bucket, object_name, created_at, size_bytes, num_rows=None, 
                 num_columns=None, schema=None, parent_dataset_id=None, processing_history=None):
        self.id = id
        self.name = name
        self.description = description
        self.original_filename = original_filename
        self.storage_filename = storage_filename
        self.file_type = file_type
        self.bucket = bucket
        self.object_name = object_name
        self.created_at = created_at
        self.size_bytes = size_bytes
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.schema = schema
        self.parent_dataset_id = parent_dataset_id
        self.processing_history = processing_history or []
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'original_filename': self.original_filename,
            'storage_filename': self.storage_filename,
            'file_type': self.file_type,
            'bucket': self.bucket,
            'object_name': self.object_name,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'size_bytes': self.size_bytes,
            'num_rows': self.num_rows,
            'num_columns': self.num_columns,
            'schema': self.schema,
            'parent_dataset_id': self.parent_dataset_id,
            'processing_history': self.processing_history
        }

class ModelMetadata:
    """Class to represent model metadata"""
    
    def __init__(self, id, name, description, model_type, dataset_id, target_column, 
                 feature_columns, hyperparameters, metrics, created_at, bucket, 
                 object_name, file_size_bytes):
        self.id = id
        self.name = name
        self.description = description
        self.model_type = model_type
        self.dataset_id = dataset_id
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.created_at = created_at
        self.bucket = bucket
        self.object_name = object_name
        self.file_size_bytes = file_size_bytes
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'dataset_id': self.dataset_id,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'bucket': self.bucket,
            'object_name': self.object_name,
            'file_size_bytes': self.file_size_bytes
        }

class DeploymentMetadata:
    """Class to represent deployment metadata"""
    
    def __init__(self, id, name, description, model_id, model_type, version, status, 
                 endpoint, created_at, updated_at):
        self.id = id
        self.name = name
        self.description = description
        self.model_id = model_id
        self.model_type = model_type
        self.version = version
        self.status = status
        self.endpoint = endpoint
        self.created_at = created_at
        self.updated_at = updated_at
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'model_id': self.model_id,
            'model_type': self.model_type,
            'version': self.version,
            'status': self.status,
            'endpoint': self.endpoint,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }

class PreprocessingJob:
    """Class to represent a preprocessing job"""
    
    def __init__(self, id, dataset_id, operations, output_name, status, 
                 created_at, updated_at, result=None, error=None):
        self.id = id
        self.dataset_id = dataset_id
        self.operations = operations
        self.output_name = output_name
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
        self.result = result
        self.error = error
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'operations': self.operations,
            'output_name': self.output_name,
            'status': self.status,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            'result': self.result,
            'error': self.error
        }

class FeatureEngineeringJob:
    """Class to represent a feature engineering job"""
    
    def __init__(self, id, dataset_id, operations, output_name, status, 
                 created_at, updated_at, result=None, error=None):
        self.id = id
        self.dataset_id = dataset_id
        self.operations = operations
        self.output_name = output_name
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
        self.result = result
        self.error = error
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'operations': self.operations,
            'output_name': self.output_name,
            'status': self.status,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            'result': self.result,
            'error': self.error
        }

class ModelTrainingJob:
    """Class to represent a model training job"""
    
    def __init__(self, id, model_id, dataset_id, model_name, hyperparameters, 
                 target_column, feature_columns, status, created_at, updated_at, 
                 result=None, error=None):
        self.id = id
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
        self.result = result
        self.error = error
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'status': self.status,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            'result': self.result,
            'error': self.error
        }

class VisualizationMetadata:
    """Class to represent visualization metadata"""
    
    def __init__(self, id, name, visualization_type, dataset_id=None, model_id=None, 
                 created_at=None, bucket=None, object_name=None):
        self.id = id
        self.name = name
        self.visualization_type = visualization_type
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.created_at = created_at or datetime.now()
        self.bucket = bucket
        self.object_name = object_name
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'visualization_type': self.visualization_type,
            'dataset_id': self.dataset_id,
            'model_id': self.model_id,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'bucket': self.bucket,
            'object_name': self.object_name
        }
