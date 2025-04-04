import os
import uuid
import logging
import json
from datetime import datetime
from flask import current_app
from services.minio_service import MinioService
from services.ml_service import MLService
from services.spark_service import SparkService
from models.metadata import ModelMetadata, DeploymentMetadata, ModelTrainingJob

# Configure logging
logger = logging.getLogger(__name__)

class ModelController:
    def __init__(self):
        self.minio_service = MinioService()
        self.ml_service = MLService()
        self.spark_service = SparkService()
        
        # In-memory storage for model metadata (replace with database in production)
        self.models = {}
        self.deployments = {}
        self.training_jobs = {}
    
    def get_available_models(self):
        """Get list of available ML models for training"""
        return current_app.config['AVAILABLE_MODELS']
    
    def recommend_model(self, dataset_id, target_column, problem_type, evaluation_metric=None):
        """
        Recommend a model based on dataset characteristics
        
        Args:
            dataset_id: ID of the dataset
            target_column: Target column name
            problem_type: Type of problem (classification/regression)
            evaluation_metric: Preferred evaluation metric
            
        Returns:
            Dict with model recommendations
        """
        try:
            # Get dataset info from DataController (simplified for this example)
            from controllers.data_controller import DataController
            data_controller = DataController()
            
            dataset_info = data_controller.get_dataset_info(dataset_id)
            if not dataset_info:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Get a preview of the data
            preview = data_controller.get_dataset_preview(dataset_id, limit=100)
            
            # Download dataset for analysis
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"model_rec_{dataset_id}.{dataset_info['file_type']}")
            
            self.minio_service.download_file(
                dataset_info['bucket'], 
                dataset_info['object_name'], 
                temp_file_path
            )
            
            # Get model recommendations
            recommendations = self.ml_service.recommend_models(
                temp_file_path,
                dataset_info['file_type'],
                target_column,
                problem_type,
                evaluation_metric
            )
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            return {
                "dataset_id": dataset_id,
                "problem_type": problem_type,
                "target_column": target_column,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error recommending model: {str(e)}")
            raise
    
    def get_model_hyperparameters(self, model_name):
        """Get hyperparameters for a specific model"""
        try:
            return self.ml_service.get_model_hyperparameters(model_name)
        except Exception as e:
            logger.error(f"Error getting model hyperparameters: {str(e)}")
            return None
    
    def compare_models(self, dataset_id, target_column, problem_type, models, evaluation_metrics, cv_folds=5):
        """
        Compare multiple models on a dataset
        
        Args:
            dataset_id: ID of the dataset
            target_column: Target column name
            problem_type: Type of problem (classification/regression)
            models: List of models to compare
            evaluation_metrics: List of evaluation metrics to use
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dict with comparison results
        """
        try:
            # Get dataset info
            from controllers.data_controller import DataController
            data_controller = DataController()
            
            dataset_info = data_controller.get_dataset_info(dataset_id)
            if not dataset_info:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Download dataset for analysis
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"model_compare_{dataset_id}.{dataset_info['file_type']}")
            
            self.minio_service.download_file(
                dataset_info['bucket'], 
                dataset_info['object_name'], 
                temp_file_path
            )
            
            # Compare models
            comparison_results = self.ml_service.compare_models(
                temp_file_path,
                dataset_info['file_type'],
                target_column,
                problem_type,
                models,
                evaluation_metrics,
                cv_folds
            )
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            return {
                "dataset_id": dataset_id,
                "problem_type": problem_type,
                "target_column": target_column,
                "models_compared": len(models),
                "evaluation_metrics": evaluation_metrics,
                "results": comparison_results
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def train_model(self, dataset_id, target_column, feature_columns, model_name, 
                    hyperparameters, test_size=0.2, random_state=42, model_description=''):
        """
        Train a machine learning model
        
        Args:
            dataset_id: ID of the dataset
            target_column: Target column name
            feature_columns: List of feature columns (None for all)
            model_name: Name of the model to train
            hyperparameters: Dict of hyperparameters
            test_size: Test set size (0-1)
            random_state: Random seed
            model_description: Description of the model
            
        Returns:
            Dict with training job information
        """
        try:
            # Generate a job ID and model ID
            job_id = str(uuid.uuid4())
            model_id = str(uuid.uuid4())
            
            # Get dataset info
            from controllers.data_controller import DataController
            data_controller = DataController()
            
            dataset_info = data_controller.get_dataset_info(dataset_id)
            if not dataset_info:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Create a new training job
            job = ModelTrainingJob(
                id=job_id,
                model_id=model_id,
                dataset_id=dataset_id,
                model_name=model_name,
                hyperparameters=hyperparameters,
                target_column=target_column,
                feature_columns=feature_columns,
                status="PENDING",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store job metadata
            self.training_jobs[job_id] = job.to_dict()
            
            # Download dataset for training
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"train_{dataset_id}.{dataset_info['file_type']}")
            temp_model_path = os.path.join(temp_dir, f"model_{model_id}.pkl")
            
            self.minio_service.download_file(
                dataset_info['bucket'], 
                dataset_info['object_name'], 
                temp_file_path
            )
            
            # Update job status
            self.training_jobs[job_id]['status'] = "RUNNING"
            self.training_jobs[job_id]['updated_at'] = datetime.now()
            
            # Train the model
            training_result = self.ml_service.train_model(
                temp_file_path,
                dataset_info['file_type'],
                target_column,
                feature_columns,
                model_name,
                hyperparameters,
                temp_model_path,
                test_size,
                random_state
            )
            
            # Upload model to MinIO
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_id}_{timestamp}.pkl"
            
            bucket_name = current_app.config['BUCKET_MODELS']
            self.minio_service.upload_file(bucket_name, model_filename, temp_model_path)
            
            # Create model metadata
            model_metadata = ModelMetadata(
                id=model_id,
                name=f"{model_name} - {timestamp}",
                description=model_description,
                model_type=model_name,
                dataset_id=dataset_id,
                target_column=target_column,
                feature_columns=feature_columns if feature_columns else training_result.get('feature_columns', []),
                hyperparameters=hyperparameters,
                metrics=training_result.get('metrics', {}),
                created_at=datetime.now(),
                bucket=bucket_name,
                object_name=model_filename,
                file_size_bytes=os.path.getsize(temp_model_path)
            )
            
            # Store model metadata
            self.models[model_id] = model_metadata.to_dict()
            
            # Update job status
            self.training_jobs[job_id]['status'] = "COMPLETED"
            self.training_jobs[job_id]['updated_at'] = datetime.now()
            self.training_jobs[job_id]['result'] = {
                "model_id": model_id,
                "metrics": training_result.get('metrics', {})
            }
            
            # Generate evaluation report and feature importance plot if available
            if 'feature_importance' in training_result:
                # Save feature importance plot
                plot_path = os.path.join(temp_dir, f"feat_imp_{model_id}.png")
                self.ml_service.plot_feature_importance(
                    training_result['feature_importance'],
                    plot_path
                )
                
                # Upload to MinIO
                plot_filename = f"{model_id}_feature_importance.png"
                bucket_name = current_app.config['BUCKET_VISUALIZATIONS']
                self.minio_service.upload_file(bucket_name, plot_filename, plot_path)
                
                # Update model metadata
                self.models[model_id]['visualizations'] = {
                    'feature_importance': {
                        'bucket': bucket_name,
                        'object_name': plot_filename
                    }
                }
                
                # Clean up plot file
                os.remove(plot_path)
            
            # Clean up temp files
            os.remove(temp_file_path)
            os.remove(temp_model_path)
            
            return {
                "job_id": job_id,
                "model_id": model_id,
                "status": "COMPLETED",
                "message": "Model training completed successfully",
                "metrics": training_result.get('metrics', {})
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            
            # Update job status if a job was created
            if 'job_id' in locals() and job_id in self.training_jobs:
                self.training_jobs[job_id]['status'] = "FAILED"
                self.training_jobs[job_id]['updated_at'] = datetime.now()
                self.training_jobs[job_id]['error'] = str(e)
            
            raise
    
    def get_model_evaluation(self, model_id):
        """Get evaluation metrics for a trained model"""
        if model_id not in self.models:
            return None
            
        try:
            model_metadata = self.models[model_id]
            return {
                "model_id": model_id,
                "model_type": model_metadata['model_type'],
                "metrics": model_metadata.get('metrics', {})
            }
        except Exception as e:
            logger.error(f"Error getting model evaluation: {str(e)}")
            return None
    
    def list_trained_models(self):
        """List all trained models"""
        return list(self.models.values())
    
    def get_model_details(self, model_id):
        """Get details of a specific trained model"""
        return self.models.get(model_id)
    
    def delete_model(self, model_id):
        """Delete a trained model"""
        if model_id not in self.models:
            return False
        
        try:
            # Get model metadata
            metadata = self.models[model_id]
            
            # Delete the object from MinIO
            self.minio_service.delete_file(metadata['bucket'], metadata['object_name'])
            
            # Delete any associated visualizations
            if 'visualizations' in metadata:
                for viz_type, viz_info in metadata['visualizations'].items():
                    self.minio_service.delete_file(viz_info['bucket'], viz_info['object_name'])
            
            # Remove from metadata store
            del self.models[model_id]
            
            # Also check if the model is deployed and undeploy if needed
            for deployment_id, deployment in list(self.deployments.items()):
                if deployment['model_id'] == model_id:
                    self.undeploy_model(deployment_id)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False
    
    def cross_validate_model(self, dataset_id, target_column, feature_columns, model_name, 
                           hyperparameters, cv_folds, evaluation_metrics, random_state=42):
        """
        Perform cross-validation for a model
        
        Args:
            dataset_id: ID of the dataset
            target_column: Target column name
            feature_columns: List of feature columns (None for all)
            model_name: Name of the model to validate
            hyperparameters: Dict of hyperparameters
            cv_folds: Number of cross-validation folds
            evaluation_metrics: List of evaluation metrics
            random_state: Random seed
            
        Returns:
            Dict with cross-validation results
        """
        try:
            # Get dataset info
            from controllers.data_controller import DataController
            data_controller = DataController()
            
            dataset_info = data_controller.get_dataset_info(dataset_id)
            if not dataset_info:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Download dataset for cross-validation
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"cv_{dataset_id}.{dataset_info['file_type']}")
            
            self.minio_service.download_file(
                dataset_info['bucket'], 
                dataset_info['object_name'], 
                temp_file_path
            )
            
            # Perform cross-validation
            cv_results = self.ml_service.cross_validate_model(
                temp_file_path,
                dataset_info['file_type'],
                target_column,
                feature_columns,
                model_name,
                hyperparameters,
                cv_folds,
                evaluation_metrics,
                random_state
            )
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            return {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "target_column": target_column,
                "cv_folds": cv_folds,
                "results": cv_results
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def get_training_job_status(self, job_id):
        """Get the status of a model training job"""
        return self.training_jobs.get(job_id)
    
    def deploy_model(self, model_id, deployment_name, description='', version='1.0'):
        """
        Deploy a trained ML model for predictions
        
        Args:
            model_id: ID of the model to deploy
            deployment_name: Name for the deployment
            description: Description of the deployment
            version: Version string
            
        Returns:
            Dict with deployment information
        """
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found")
        
        try:
            # Generate a deployment ID
            deployment_id = str(uuid.uuid4())
            
            # Get model metadata
            model_metadata = self.models[model_id]
            
            # Create deployment metadata
            deployment = DeploymentMetadata(
                id=deployment_id,
                name=deployment_name,
                description=description,
                model_id=model_id,
                model_type=model_metadata['model_type'],
                version=version,
                status="ACTIVE",
                endpoint=f"/deployment/{deployment_id}/predict",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store deployment metadata
            self.deployments[deployment_id] = deployment.to_dict()
            
            # Get the model from MinIO and load it in memory
            # In a production system, we might use a separate service for model serving
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_model_path = os.path.join(temp_dir, f"deploy_{model_id}.pkl")
            
            self.minio_service.download_file(
                model_metadata['bucket'], 
                model_metadata['object_name'], 
                temp_model_path
            )
            
            # Load the model
            self.ml_service.load_model(deployment_id, temp_model_path)
            
            # Clean up temp file
            os.remove(temp_model_path)
            
            return {
                "deployment_id": deployment_id,
                "model_id": model_id,
                "name": deployment_name,
                "status": "ACTIVE",
                "endpoint": f"/deployment/{deployment_id}/predict",
                "message": "Model deployed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise
    
    def list_deployments(self):
        """List all deployed models"""
        return list(self.deployments.values())
    
    def get_deployment(self, deployment_id):
        """Get details of a specific deployment"""
        return self.deployments.get(deployment_id)
    
    def undeploy_model(self, deployment_id):
        """Undeploy a model"""
        if deployment_id not in self.deployments:
            return False
        
        try:
            # Remove from loaded models
            self.ml_service.unload_model(deployment_id)
            
            # Update deployment status
            self.deployments[deployment_id]['status'] = "INACTIVE"
            self.deployments[deployment_id]['updated_at'] = datetime.now()
            
            # Remove from deployments
            del self.deployments[deployment_id]
            
            return True
        except Exception as e:
            logger.error(f"Error undeploying model: {str(e)}")
            return False
    
    def predict(self, deployment_id, data):
        """
        Make predictions using a deployed model
        
        Args:
            deployment_id: ID of the deployment
            data: List of input data points
            
        Returns:
            List of predictions
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment with ID {deployment_id} not found")
        
        if self.deployments[deployment_id]['status'] != "ACTIVE":
            raise ValueError(f"Deployment {deployment_id} is not active")
        
        try:
            # Make predictions
            predictions = self.ml_service.predict(deployment_id, data)
            
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def batch_predict(self, deployment_id, dataset_id, output_name):
        """
        Make batch predictions using a deployed model and a dataset
        
        Args:
            deployment_id: ID of the deployment
            dataset_id: ID of the dataset
            output_name: Name for the output dataset
            
        Returns:
            Dict with batch prediction results
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment with ID {deployment_id} not found")
        
        if self.deployments[deployment_id]['status'] != "ACTIVE":
            raise ValueError(f"Deployment {deployment_id} is not active")
        
        try:
            # Get dataset info
            from controllers.data_controller import DataController
            data_controller = DataController()
            
            dataset_info = data_controller.get_dataset_info(dataset_id)
            if not dataset_info:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Download dataset for predictions
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_input_path = os.path.join(temp_dir, f"pred_in_{dataset_id}.{dataset_info['file_type']}")
            temp_output_path = os.path.join(temp_dir, f"pred_out_{dataset_id}.{dataset_info['file_type']}")
            
            self.minio_service.download_file(
                dataset_info['bucket'], 
                dataset_info['object_name'], 
                temp_input_path
            )
            
            # Get the model
            deployment = self.deployments[deployment_id]
            model_id = deployment['model_id']
            model_metadata = self.models[model_id]
            
            # Make batch predictions
            prediction_result = self.ml_service.batch_predict(
                deployment_id,
                temp_input_path,
                temp_output_path,
                dataset_info['file_type'],
                model_metadata['feature_columns']
            )
            
            # Upload results to MinIO
            result_dataset_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_filename = f"{result_dataset_id}_{timestamp}_predictions.{dataset_info['file_type']}"
            
            bucket_name = current_app.config['BUCKET_DATA_PROCESSED']
            self.minio_service.upload_file(bucket_name, result_filename, temp_output_path)
            
            # Get data preview and schema of results
            preview_data, schema = self.spark_service.get_data_preview_and_schema(
                temp_output_path, 
                dataset_info['file_type']
            )
            
            # Create metadata for result dataset
            from models.metadata import DatasetMetadata
            result_metadata = DatasetMetadata(
                id=result_dataset_id,
                name=output_name,
                description=f"Predictions from {dataset_info['name']} using model {model_metadata['name']}",
                original_filename=dataset_info['original_filename'],
                storage_filename=result_filename,
                file_type=dataset_info['file_type'],
                bucket=bucket_name,
                object_name=result_filename,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(temp_output_path),
                num_rows=preview_data.get('num_rows', 0),
                num_columns=preview_data.get('num_columns', 0),
                schema=schema,
                parent_dataset_id=dataset_id,
                processing_history=[{
                    "type": "prediction",
                    "model_id": model_id,
                    "deployment_id": deployment_id
                }]
            )
            
            # Store result dataset metadata
            data_controller.datasets[result_dataset_id] = result_metadata.to_dict()
            
            # Clean up temp files
            os.remove(temp_input_path)
            os.remove(temp_output_path)
            
            return {
                "result_dataset_id": result_dataset_id,
                "rows_processed": prediction_result.get('rows_processed', 0),
                "message": "Batch prediction completed successfully",
                "preview": preview_data
            }
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def get_deployment_metadata(self, deployment_id):
        """Get metadata for a deployed model"""
        if deployment_id not in self.deployments:
            return None
            
        try:
            deployment = self.deployments[deployment_id]
            model_id = deployment['model_id']
            
            if model_id in self.models:
                model = self.models[model_id]
                
                return {
                    "deployment_id": deployment_id,
                    "deployment_name": deployment['name'],
                    "status": deployment['status'],
                    "model_id": model_id,
                    "model_type": model['model_type'],
                    "feature_columns": model['feature_columns'],
                    "target_column": model['target_column'],
                    "metrics": model.get('metrics', {}),
                    "created_at": deployment['created_at'].isoformat() if isinstance(deployment['created_at'], datetime) else deployment['created_at']
                }
            else:
                return {
                    "deployment_id": deployment_id,
                    "deployment_name": deployment['name'],
                    "status": deployment['status'],
                    "model_id": model_id,
                    "error": "Model metadata not found"
                }
                
        except Exception as e:
            logger.error(f"Error getting deployment metadata: {str(e)}")
            return None
