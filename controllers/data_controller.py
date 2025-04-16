import os
import uuid
import logging
import json
from datetime import datetime
from flask import current_app
from werkzeug.utils import secure_filename
from models.metadata import DatasetMetadata, PreprocessingJob, FeatureEngineeringJob

# Configure logging
logger = logging.getLogger(__name__)

class DataController:
    def __init__(self):
        # Initialize services with error handling
        self.minio_service = self._initialize_minio_service()
        self.spark_service = self._initialize_spark_service()
        
        # In-memory storage for dataset metadata (replace with database in production)
        self.datasets = {}
        self.preprocessing_jobs = {}
        self.feature_engineering_jobs = {}
        
        # Ensure MinIO buckets exist if service is available
        if self.minio_service:
            self._ensure_buckets_exist()
    
    def _initialize_minio_service(self):
        """Initialize MinIO service with fallback to mock service if needed"""
        # First try to check if we should always use mock services
        try:
            from flask import current_app
            use_mock = current_app.config.get('USE_MOCK_SERVICES', False)
            if use_mock:
                from services.mock_minio_service import MockMinioService
                logger.info("Using MockMinioService as configured")
                return MockMinioService()
        except (RuntimeError, ImportError):
            # Outside Flask context or module import error, continue with normal flow
            pass
            
        # Try to initialize the real MinIO service
        try:
            from services.minio_service import MinioService
            service = MinioService()
            # Test connection by listing buckets
            service.client.list_buckets()
            logger.info("Successfully connected to MinIO service")
            return service
        except Exception as e:
            logger.warning(f"Could not initialize MinIO service: {str(e)}")
            try:
                from services.mock_minio_service import MockMinioService
                logger.info("Using MockMinioService as fallback")
                return MockMinioService()
            except Exception as mock_e:
                logger.error(f"Could not initialize mock MinIO service: {str(mock_e)}")
                return None
    
    def _initialize_spark_service(self):
        """Initialize Spark service with graceful fallback if needed"""
        try:
            from services.spark_service import SparkService
            return SparkService()
        except Exception as e:
            logger.warning(f"Could not initialize Spark service: {str(e)}")
            try:
                from services.mock_spark_service import MockSparkService
                logger.info("Using MockSparkService as fallback")
                return MockSparkService()
            except Exception as mock_e:
                logger.error(f"Could not initialize mock Spark service: {str(mock_e)}")
                return None
    
    def _ensure_buckets_exist(self):
        """Ensure all required MinIO buckets exist"""
        if not self.minio_service:
            logger.warning("MinIO service not available, skipping bucket creation")
            return
            
        # Using default bucket names if config is not available
        buckets = [
            current_app.config.get('BUCKET_DATA_RAW', 'raw-data'),
            current_app.config.get('BUCKET_DATA_PROCESSED', 'processed-data'),
            current_app.config.get('BUCKET_MODELS', 'models'),
            current_app.config.get('BUCKET_VISUALIZATIONS', 'visualizations'),
            current_app.config.get('BUCKET_EVALUATIONS', 'evaluations')
        ]
        
        for bucket in buckets:
            self.minio_service.ensure_bucket_exists(bucket)
    
    def upload_dataset(self, file, dataset_name, description, file_ext):
        """
        Upload a dataset file to MinIO
        
        Args:
            file: The file object to upload
            dataset_name: Name of the dataset
            description: Description of the dataset
            file_ext: File extension (csv, parquet, json)
            
        Returns:
            Dict with dataset metadata
        """
        try:
            # Check if services are available
            if not self.minio_service:
                raise RuntimeError("MinIO service is not available")
            
            if not self.spark_service:
                raise RuntimeError("Spark service is not available")
                
            # Generate a unique dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Secure the filename
            original_filename = secure_filename(file.filename)
            
            # Create a structured filename for storage
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            storage_filename = f"{dataset_id}_{timestamp}.{file_ext}"
            
            # Save the file to a temporary location
            temp_dir = current_app.config.get('TEMP_DIR', '/tmp/ml-pipeline')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, storage_filename)
            file.save(temp_file_path)
            
            # Upload the file to MinIO
            bucket_name = current_app.config.get('BUCKET_DATA_RAW', 'raw-data')
            object_name = storage_filename
            self.minio_service.upload_file(bucket_name, object_name, temp_file_path)
            
            # Get data preview and schema using Spark
            preview_data, schema = self.spark_service.get_data_preview_and_schema(temp_file_path, file_ext)
            
            # Create dataset metadata
            metadata = DatasetMetadata(
                id=dataset_id,
                name=dataset_name,
                description=description,
                original_filename=original_filename,
                storage_filename=storage_filename,
                file_type=file_ext,
                bucket=bucket_name,
                object_name=object_name,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(temp_file_path),
                num_rows=preview_data.get('num_rows', 0),
                num_columns=preview_data.get('num_columns', 0),
                schema=schema
            )
            
            # Store metadata (in-memory for now, would be database in production)
            self.datasets[dataset_id] = metadata.to_dict()
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
            return {
                "dataset_id": dataset_id,
                "name": dataset_name,
                "message": "Dataset uploaded successfully",
                "preview": preview_data
            }
            
        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            # Make sure temp file is cleaned up
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            raise
    
    def list_datasets(self):
        """List all available datasets"""
        return list(self.datasets.values())
    
    def get_dataset_info(self, dataset_id):
        """Get information about a specific dataset"""
        return self.datasets.get(dataset_id)
    
    def delete_dataset(self, dataset_id):
        """Delete a dataset"""
        if dataset_id not in self.datasets:
            return False
        
        try:
            # Get dataset metadata
            metadata = self.datasets[dataset_id]
            
            # Delete the object from MinIO
            self.minio_service.delete_file(metadata['bucket'], metadata['object_name'])
            
            # Remove from metadata store
            del self.datasets[dataset_id]
            
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset: {str(e)}")
            return False
    
    def preprocess_dataset(self, dataset_id, operations, output_name):
        """
        Perform data preprocessing operations
        
        Args:
            dataset_id: ID of the dataset to preprocess
            operations: List of preprocessing operations
            output_name: Name for the processed dataset
            
        Returns:
            Dict with job information
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Get dataset metadata
        dataset_metadata = self.datasets[dataset_id]
        
        # Create a new preprocessing job
        job = PreprocessingJob(
            id=job_id,
            dataset_id=dataset_id,
            operations=operations,
            output_name=output_name,
            status="PENDING",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store job metadata
        self.preprocessing_jobs[job_id] = job.to_dict()
        
        # Start preprocessing in background (would use task queue in production)
        try:
            # Update job status
            self.preprocessing_jobs[job_id]['status'] = "RUNNING"
            self.preprocessing_jobs[job_id]['updated_at'] = datetime.now()
            
            # Get source file from MinIO
            source_bucket = dataset_metadata['bucket']
            source_object = dataset_metadata['object_name']
            file_type = dataset_metadata['file_type']
            
            # Download to temp location
            temp_dir = current_app.config.get("TEMP_DIR", "/tmp/ml-pipeline")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_input_path = os.path.join(temp_dir, f"input_{job_id}.{file_type}")
            temp_output_path = os.path.join(temp_dir, f"output_{job_id}.{file_type}")
            
            self.minio_service.download_file(source_bucket, source_object, temp_input_path)
            
            # Process with Spark
            result = self.spark_service.preprocess_data(
                temp_input_path, 
                temp_output_path, 
                operations, 
                file_type
            )
            
            # Upload processed file to MinIO
            processed_dataset_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_filename = f"{processed_dataset_id}_{timestamp}_processed.{file_type}"
            
            target_bucket = current_app.config.get("BUCKET_DATA_PROCESSED", "processed-data")
            self.minio_service.upload_file(target_bucket, processed_filename, temp_output_path)
            
            # Get data preview and schema of processed data
            preview_data, schema = self.spark_service.get_data_preview_and_schema(temp_output_path, file_type)
            
            # Create metadata for processed dataset
            processed_metadata = DatasetMetadata(
                id=processed_dataset_id,
                name=output_name,
                description=f"Processed from {dataset_metadata['name']} with {len(operations)} operations",
                original_filename=dataset_metadata['original_filename'],
                storage_filename=processed_filename,
                file_type=file_type,
                bucket=target_bucket,
                object_name=processed_filename,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(temp_output_path),
                num_rows=preview_data.get('num_rows', 0),
                num_columns=preview_data.get('num_columns', 0),
                schema=schema,
                parent_dataset_id=dataset_id,
                processing_history=operations
            )
            
            # Store processed dataset metadata
            self.datasets[processed_dataset_id] = processed_metadata.to_dict()
            
            # Update job status
            self.preprocessing_jobs[job_id]['status'] = "COMPLETED"
            self.preprocessing_jobs[job_id]['updated_at'] = datetime.now()
            self.preprocessing_jobs[job_id]['result'] = {
                "processed_dataset_id": processed_dataset_id,
                "operations_performed": len(operations),
                "rows_processed": preview_data.get('num_rows', 0)
            }
            
            # Clean up temp files
            os.remove(temp_input_path)
            os.remove(temp_output_path)
            
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "processed_dataset_id": processed_dataset_id,
                "message": "Data preprocessing completed successfully",
                "preview": preview_data
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing job {job_id}: {str(e)}")
            
            # Update job status
            self.preprocessing_jobs[job_id]['status'] = "FAILED"
            self.preprocessing_jobs[job_id]['updated_at'] = datetime.now()
            self.preprocessing_jobs[job_id]['error'] = str(e)
            
            raise
    
    def get_preprocessing_job_status(self, job_id):
        """Get the status of a preprocessing job"""
        return self.preprocessing_jobs.get(job_id)
    
    def get_dataset_preview(self, dataset_id, limit=10):
        """Get a preview of the dataset (first few rows)"""
        if dataset_id not in self.datasets:
            return None
        
        try:
            # Get dataset metadata
            metadata = self.datasets[dataset_id]
            
            # Download from MinIO to temp location
            temp_dir = current_app.config.get("TEMP_DIR", "/tmp/ml-pipeline")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"preview_{dataset_id}.{metadata['file_type']}")
            
            self.minio_service.download_file(
                metadata['bucket'], 
                metadata['object_name'], 
                temp_file_path
            )
            
            # Get preview using Spark
            preview_data, _ = self.spark_service.get_data_preview_and_schema(
                temp_file_path, 
                metadata['file_type'], 
                limit
            )
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Error getting dataset preview: {str(e)}")
            return None
    
    def get_dataset_schema(self, dataset_id):
        """Get the schema of the dataset"""
        if dataset_id not in self.datasets:
            return None
            
        try:
            return self.datasets[dataset_id]['schema']
        except Exception as e:
            logger.error(f"Error getting dataset schema: {str(e)}")
            return None
    
    def engineer_features(self, dataset_id, operations, output_name):
        """
        Perform feature engineering operations
        
        Args:
            dataset_id: ID of the dataset
            operations: List of feature engineering operations
            output_name: Name for the output dataset
            
        Returns:
            Dict with job information
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Get dataset metadata
        dataset_metadata = self.datasets[dataset_id]
        
        # Create a new feature engineering job
        job = FeatureEngineeringJob(
            id=job_id,
            dataset_id=dataset_id,
            operations=operations,
            output_name=output_name,
            status="PENDING",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store job metadata
        self.feature_engineering_jobs[job_id] = job.to_dict()
        
        # Start feature engineering in background
        try:
            # Update job status
            self.feature_engineering_jobs[job_id]['status'] = "RUNNING"
            self.feature_engineering_jobs[job_id]['updated_at'] = datetime.now()
            
            # Get source file from MinIO
            source_bucket = dataset_metadata['bucket']
            source_object = dataset_metadata['object_name']
            file_type = dataset_metadata['file_type']
            
            # Download to temp location
            temp_dir = current_app.config.get("TEMP_DIR", "/tmp/ml-pipeline")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_input_path = os.path.join(temp_dir, f"input_{job_id}.{file_type}")
            temp_output_path = os.path.join(temp_dir, f"output_{job_id}.{file_type}")
            
            self.minio_service.download_file(source_bucket, source_object, temp_input_path)
            
            # Process with Spark
            result = self.spark_service.engineer_features(
                temp_input_path, 
                temp_output_path, 
                operations, 
                file_type
            )
            
            # Upload processed file to MinIO
            processed_dataset_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_filename = f"{processed_dataset_id}_{timestamp}_featured.{file_type}"
            
            target_bucket = current_app.config.get("BUCKET_DATA_PROCESSED", "processed-data")
            self.minio_service.upload_file(target_bucket, processed_filename, temp_output_path)
            
            # Get data preview and schema of processed data
            preview_data, schema = self.spark_service.get_data_preview_and_schema(temp_output_path, file_type)
            
            # Create metadata for processed dataset
            processed_metadata = DatasetMetadata(
                id=processed_dataset_id,
                name=output_name,
                description=f"Feature engineered from {dataset_metadata['name']} with {len(operations)} operations",
                original_filename=dataset_metadata['original_filename'],
                storage_filename=processed_filename,
                file_type=file_type,
                bucket=target_bucket,
                object_name=processed_filename,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(temp_output_path),
                num_rows=preview_data.get('num_rows', 0),
                num_columns=preview_data.get('num_columns', 0),
                schema=schema,
                parent_dataset_id=dataset_id,
                processing_history=operations
            )
            
            # Store processed dataset metadata
            self.datasets[processed_dataset_id] = processed_metadata.to_dict()
            
            # Update job status
            self.feature_engineering_jobs[job_id]['status'] = "COMPLETED"
            self.feature_engineering_jobs[job_id]['updated_at'] = datetime.now()
            self.feature_engineering_jobs[job_id]['result'] = {
                "processed_dataset_id": processed_dataset_id,
                "operations_performed": len(operations),
                "rows_processed": preview_data.get('num_rows', 0)
            }
            
            # Clean up temp files
            os.remove(temp_input_path)
            os.remove(temp_output_path)
            
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "processed_dataset_id": processed_dataset_id,
                "message": "Feature engineering completed successfully",
                "preview": preview_data
            }
            
        except Exception as e:
            logger.error(f"Error in feature engineering job {job_id}: {str(e)}")
            
            # Update job status
            self.feature_engineering_jobs[job_id]['status'] = "FAILED"
            self.feature_engineering_jobs[job_id]['updated_at'] = datetime.now()
            self.feature_engineering_jobs[job_id]['error'] = str(e)
            
            raise
    
    def get_feature_engineering_job_status(self, job_id):
        """Get the status of a feature engineering job"""
        return self.feature_engineering_jobs.get(job_id)
    
    def get_dataset_features(self, dataset_id):
        """Get the list of features in the dataset"""
        if dataset_id not in self.datasets:
            return None
            
        try:
            schema = self.datasets[dataset_id]['schema']
            features = {
                "numerical_features": [],
                "categorical_features": [],
                "datetime_features": [],
                "text_features": [],
                "all_features": []
            }
            
            for field in schema.get('fields', []):
                field_name = field.get('name')
                field_type = field.get('type')
                
                features['all_features'].append(field_name)
                
                # Categorize features by type
                if field_type in ['int', 'long', 'float', 'double']:
                    features['numerical_features'].append(field_name)
                elif field_type in ['string']:
                    # Simple heuristic: if it has few unique values, consider it categorical
                    # In a real system we'd want better logic here
                    features['categorical_features'].append(field_name)
                    # Check if it might be text data or datetime
                    # This is a simplified approach
                elif field_type in ['date', 'timestamp']:
                    features['datetime_features'].append(field_name)
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting dataset features: {str(e)}")
            return None
    
    def calculate_feature_importance(self, dataset_id, target_column, method='mutual_info'):
        """
        Calculate feature importance
        
        Args:
            dataset_id: ID of the dataset
            target_column: Target column name
            method: Method to use for feature importance calculation
            
        Returns:
            Dict with feature importance values
        """
        if dataset_id not in self.datasets:
            return None
            
        try:
            # Get dataset metadata
            metadata = self.datasets[dataset_id]
            
            # Download from MinIO to temp location
            temp_dir = current_app.config.get("TEMP_DIR", "/tmp/ml-pipeline")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"feat_imp_{dataset_id}.{metadata['file_type']}")
            
            self.minio_service.download_file(
                metadata['bucket'], 
                metadata['object_name'], 
                temp_file_path
            )
            
            # Calculate feature importance
            feature_importance = self.spark_service.calculate_feature_importance(
                temp_file_path,
                metadata['file_type'],
                target_column,
                method
            )
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            return {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "method": method,
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return None
