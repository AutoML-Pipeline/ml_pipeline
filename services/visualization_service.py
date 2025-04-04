import os
import uuid
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import current_app
from services.minio_service import MinioService
from models.metadata import VisualizationMetadata

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for generating and managing visualizations"""
    
    def __init__(self):
        """Initialize the visualization service"""
        self.minio_service = MinioService()
        
        # In-memory storage for visualization metadata (replace with database in production)
        self.visualizations = {}
    
    def _read_file(self, file_path, file_type):
        """
        Read a file into a pandas DataFrame
        
        Args:
            file_path: Path to the file
            file_type: Type of the file (csv, parquet, json)
            
        Returns:
            pandas DataFrame
        """
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            return df
        except Exception as e:
            logger.error(f"Error reading file {file_path} of type {file_type}: {str(e)}")
            raise
    
    def _get_dataset_file(self, dataset_id):
        """
        Get the file path for a dataset
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Tuple of (temp_file_path, file_type, dataset_info)
        """
        # Get dataset info from DataController
        from controllers.data_controller import DataController
        data_controller = DataController()
        
        dataset_info = data_controller.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Download from MinIO to temp location
        temp_dir = current_app.config['TEMP_DIR']
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, f"viz_{dataset_id}.{dataset_info['file_type']}")
        
        self.minio_service.download_file(
            dataset_info['bucket'], 
            dataset_info['object_name'], 
            temp_file_path
        )
        
        return temp_file_path, dataset_info['file_type'], dataset_info
    
    def _get_model_file(self, model_id):
        """
        Get the file path for a model
        
        Args:
            model_id: ID of the model
            
        Returns:
            Tuple of (temp_file_path, model_info)
        """
        # Get model info from ModelController
        from controllers.model_controller import ModelController
        model_controller = ModelController()
        
        model_info = model_controller.get_model_details(model_id)
        if not model_info:
            raise ValueError(f"Model with ID {model_id} not found")
        
        # Download from MinIO to temp location
        temp_dir = current_app.config['TEMP_DIR']
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, f"model_{model_id}.pkl")
        
        self.minio_service.download_file(
            model_info['bucket'], 
            model_info['object_name'], 
            temp_file_path
        )
        
        return temp_file_path, model_info
    
    def create_data_visualization(self, dataset_id, visualization_type, columns, title, options=None):
        """
        Generate data visualizations
        
        Args:
            dataset_id: ID of the dataset
            visualization_type: Type of visualization (histogram, scatter, etc.)
            columns: List of columns to include in the visualization
            title: Title for the visualization
            options: Additional options for the visualization
            
        Returns:
            Dict with visualization information
        """
        try:
            # Generate a visualization ID
            viz_id = str(uuid.uuid4())
            
            # Get dataset file
            temp_file_path, file_type, dataset_info = self._get_dataset_file(dataset_id)
            
            # Read the data
            df = self._read_file(temp_file_path, file_type)
            
            # Verify columns exist
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columns not found in dataset: {missing_columns}")
            
            # Set default options
            options = options or {}
            figsize = options.get('figsize', [10, 6])
            
            # Import matplotlib
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("darkgrid")
            
            # Create figure
            plt.figure(figsize=(figsize[0], figsize[1]))
            
            # Generate visualization based on type
            if visualization_type == 'histogram':
                if len(columns) != 1:
                    raise ValueError("Histogram requires exactly one column")
                
                column = columns[0]
                bins = options.get('bins', 20)
                kde = options.get('kde', True)
                
                ax = sns.histplot(df[column], bins=bins, kde=kde)
                
                # Add labels
                plt.xlabel(options.get('x_label', column))
                plt.ylabel(options.get('y_label', 'Frequency'))
            
            elif visualization_type == 'scatter':
                if len(columns) != 2:
                    raise ValueError("Scatter plot requires exactly two columns")
                
                x_col, y_col = columns
                
                # Optional hue column
                hue_col = options.get('hue')
                
                if hue_col:
                    ax = sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df)
                else:
                    ax = sns.scatterplot(x=x_col, y=y_col, data=df)
                
                # Add labels
                plt.xlabel(options.get('x_label', x_col))
                plt.ylabel(options.get('y_label', y_col))
            
            elif visualization_type == 'bar':
                if len(columns) != 2:
                    raise ValueError("Bar plot requires exactly two columns")
                
                x_col, y_col = columns
                
                # Aggregate if needed
                if options.get('aggregate', True):
                    aggfunc = options.get('aggfunc', 'mean')
                    agg_df = df.groupby(x_col)[y_col].agg(aggfunc).reset_index()
                    ax = sns.barplot(x=x_col, y=y_col, data=agg_df)
                else:
                    ax = sns.barplot(x=x_col, y=y_col, data=df)
                
                # Add labels
                plt.xlabel(options.get('x_label', x_col))
                plt.ylabel(options.get('y_label', y_col))
                
                # Rotate x-tick labels if there are many categories
                if df[x_col].nunique() > 5:
                    plt.xticks(rotation=45, ha='right')
            
            elif visualization_type == 'box':
                # Box plot can accept one or multiple columns
                if len(columns) == 1:
                    ax = sns.boxplot(y=columns[0], data=df)
                    plt.ylabel(options.get('y_label', columns[0]))
                else:
                    # For multiple columns, melt the dataframe
                    melt_df = df[columns].melt()
                    ax = sns.boxplot(x='variable', y='value', data=melt_df)
                    plt.xlabel(options.get('x_label', 'Variables'))
                    plt.ylabel(options.get('y_label', 'Values'))
            
            elif visualization_type == 'heatmap':
                # Compute correlation matrix
                corr_df = df[columns].corr()
                
                # Create heatmap
                cmap = options.get('cmap', 'coolwarm')
                ax = sns.heatmap(corr_df, annot=True, cmap=cmap, linewidths=0.5)
            
            elif visualization_type == 'pie':
                if len(columns) != 1:
                    raise ValueError("Pie chart requires exactly one column")
                
                column = columns[0]
                
                # Count values or use provided values
                if options.get('count', True):
                    values = df[column].value_counts()
                else:
                    if len(columns) != 2:
                        raise ValueError("Pie chart with values requires two columns")
                    values_col = columns[1]
                    values = df.groupby(column)[values_col].sum()
                
                # Create pie chart
                plt.pie(values, labels=values.index, autopct='%1.1f%%', 
                      startangle=90, shadow=options.get('shadow', False))
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            elif visualization_type == 'line':
                if len(columns) < 2:
                    raise ValueError("Line plot requires at least two columns")
                
                x_col = columns[0]
                y_cols = columns[1:]
                
                # Sort by x column
                df = df.sort_values(by=x_col)
                
                # Create line plot
                for y_col in y_cols:
                    plt.plot(df[x_col], df[y_col], label=y_col)
                
                # Add labels
                plt.xlabel(options.get('x_label', x_col))
                plt.ylabel(options.get('y_label', 'Value'))
                plt.legend()
                
                # Add grid
                plt.grid(True, alpha=0.3)
            
            else:
                raise ValueError(f"Unsupported visualization type: {visualization_type}")
            
            # Add title
            plt.title(title)
            
            # Add tight layout
            plt.tight_layout()
            
            # Save visualization to a temporary file
            temp_viz_path = os.path.join(current_app.config['TEMP_DIR'], f"viz_{viz_id}.png")
            plt.savefig(temp_viz_path, dpi=300)
            plt.close()
            
            # Upload to MinIO
            viz_filename = f"{viz_id}_{visualization_type}.png"
            bucket_name = current_app.config['BUCKET_VISUALIZATIONS']
            
            self.minio_service.ensure_bucket_exists(bucket_name)
            self.minio_service.upload_file(bucket_name, viz_filename, temp_viz_path)
            
            # Create visualization metadata
            viz_metadata = VisualizationMetadata(
                id=viz_id,
                name=title,
                visualization_type=visualization_type,
                dataset_id=dataset_id,
                created_at=datetime.now(),
                bucket=bucket_name,
                object_name=viz_filename
            )
            
            # Store metadata
            self.visualizations[viz_id] = viz_metadata.to_dict()
            
            # Get presigned URL for the visualization
            viz_url = self.minio_service.get_object_url(bucket_name, viz_filename)
            
            # Clean up temporary files
            os.remove(temp_file_path)
            os.remove(temp_viz_path)
            
            return {
                "visualization_id": viz_id,
                "title": title,
                "type": visualization_type,
                "dataset_id": dataset_id,
                "url": viz_url
            }
            
        except Exception as e:
            logger.error(f"Error creating data visualization: {str(e)}")
            raise
    
    def create_model_visualization(self, model_id, visualization_type, title, options=None):
        """
        Generate model visualizations
        
        Args:
            model_id: ID of the model
            visualization_type: Type of visualization (confusion_matrix, roc_curve, etc.)
            title: Title for the visualization
            options: Additional options for the visualization
            
        Returns:
            Dict with visualization information
        """
        try:
            # Generate a visualization ID
            viz_id = str(uuid.uuid4())
            
            # Get model controller
            from controllers.model_controller import ModelController
            model_controller = ModelController()
            
            # Get model details
            model_details = model_controller.get_model_details(model_id)
            if not model_details:
                raise ValueError(f"Model with ID {model_id} not found")
            
            # Get dataset for the model
            dataset_id = model_details['dataset_id']
            temp_file_path, file_type, dataset_info = self._get_dataset_file(dataset_id)
            
            # Read the data
            df = self._read_file(temp_file_path, file_type)
            
            # Get model file
            temp_model_path, model_info = self._get_model_file(model_id)
            
            # Import pickle to load the model
            import pickle
            with open(temp_model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Set default options
            options = options or {}
            figsize = options.get('figsize', [10, 6])
            
            # Import matplotlib
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("darkgrid")
            
            # Create figure
            plt.figure(figsize=(figsize[0], figsize[1]))
            
            # Get target and features
            target_column = model_details['target_column']
            feature_columns = model_details['feature_columns']
            
            # Prepare data
            from services.ml_service import MLService
            ml_service = MLService()
            X, y, _ = ml_service._prepare_data(df, target_column, feature_columns)
            
            # Generate visualization based on type
            if visualization_type == 'confusion_matrix':
                # Only for classification models
                if 'classifier' not in model_details['model_type']:
                    raise ValueError("Confusion matrix is only for classification models")
                
                from sklearn.metrics import confusion_matrix
                
                # Get predictions
                y_pred = model.predict(X)
                
                # Compute confusion matrix
                cm = confusion_matrix(y, y_pred)
                
                # Plot confusion matrix
                cmap = options.get('cmap', 'Blues')
                ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
                
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
            
            elif visualization_type == 'roc_curve':
                # Only for classification models
                if 'classifier' not in model_details['model_type']:
                    raise ValueError("ROC curve is only for classification models")
                
                from sklearn.metrics import roc_curve, auc
                
                # Check if the model has predict_proba method
                if not hasattr(model, 'predict_proba'):
                    raise ValueError("Model does not support probability predictions")
                
                # Get probability predictions for the positive class
                y_prob = model.predict_proba(X)[:, 1]
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower right')
            
            elif visualization_type == 'precision_recall':
                # Only for classification models
                if 'classifier' not in model_details['model_type']:
                    raise ValueError("Precision-Recall curve is only for classification models")
                
                from sklearn.metrics import precision_recall_curve, average_precision_score
                
                # Check if the model has predict_proba method
                if not hasattr(model, 'predict_proba'):
                    raise ValueError("Model does not support probability predictions")
                
                # Get probability predictions for the positive class
                y_prob = model.predict_proba(X)[:, 1]
                
                # Compute precision-recall curve
                precision, recall, _ = precision_recall_curve(y, y_prob)
                ap = average_precision_score(y, y_prob)
                
                # Plot precision-recall curve
                plt.plot(recall, precision, label=f'AP = {ap:.3f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.legend(loc='upper right')
            
            elif visualization_type == 'feature_importance':
                # Feature importance is available in model details
                feature_importance = model_details.get('feature_importance')
                
                if not feature_importance:
                    # Try to calculate it from the model
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = [
                            {"feature": feature, "importance": float(importance)}
                            for feature, importance in zip(feature_columns, model.feature_importances_)
                        ]
                        # Sort by importance
                        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                    else:
                        raise ValueError("Feature importance not available for this model")
                
                # Extract feature names and importance values
                features = [item['feature'] for item in feature_importance[:10]]  # Top 10 features
                importances = [item['importance'] for item in feature_importance[:10]]
                
                # Create horizontal bar chart
                plt.barh(features, importances)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
            
            elif visualization_type == 'learning_curve':
                from sklearn.model_selection import learning_curve
                import numpy as np
                
                # Generate learning curve data
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y, cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10)
                )
                
                # Calculate means and standard deviations
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                # Plot learning curve
                plt.plot(train_sizes, train_mean, label='Training score')
                plt.plot(train_sizes, test_mean, label='Cross-validation score')
                
                # Add shaded regions for standard deviation
                plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
                plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
                
                plt.xlabel('Training examples')
                plt.ylabel('Score')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
            
            elif visualization_type == 'residuals':
                # Only for regression models
                if 'regressor' not in model_details['model_type']:
                    raise ValueError("Residual plot is only for regression models")
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate residuals
                residuals = y - y_pred
                
                # Create residual plot
                plt.scatter(y_pred, residuals)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Predicted values')
                plt.ylabel('Residuals')
                
                # Add trend line if requested
                if options.get('add_trend', True):
                    from scipy.stats import linregress
                    slope, intercept, _, _, _ = linregress(y_pred, residuals)
                    line_x = np.array([min(y_pred), max(y_pred)])
                    line_y = intercept + slope * line_x
                    plt.plot(line_x, line_y, 'g--', alpha=0.8)
            
            else:
                raise ValueError(f"Unsupported visualization type: {visualization_type}")
            
            # Add title
            plt.title(title)
            
            # Add tight layout
            plt.tight_layout()
            
            # Save visualization to a temporary file
            temp_viz_path = os.path.join(current_app.config['TEMP_DIR'], f"viz_{viz_id}.png")
            plt.savefig(temp_viz_path, dpi=300)
            plt.close()
            
            # Upload to MinIO
            viz_filename = f"{viz_id}_{visualization_type}.png"
            bucket_name = current_app.config['BUCKET_VISUALIZATIONS']
            
            self.minio_service.ensure_bucket_exists(bucket_name)
            self.minio_service.upload_file(bucket_name, viz_filename, temp_viz_path)
            
            # Create visualization metadata
            viz_metadata = VisualizationMetadata(
                id=viz_id,
                name=title,
                visualization_type=visualization_type,
                model_id=model_id,
                created_at=datetime.now(),
                bucket=bucket_name,
                object_name=viz_filename
            )
            
            # Store metadata
            self.visualizations[viz_id] = viz_metadata.to_dict()
            
            # Get presigned URL for the visualization
            viz_url = self.minio_service.get_object_url(bucket_name, viz_filename)
            
            # Clean up temporary files
            os.remove(temp_file_path)
            os.remove(temp_model_path)
            os.remove(temp_viz_path)
            
            return {
                "visualization_id": viz_id,
                "title": title,
                "type": visualization_type,
                "model_id": model_id,
                "url": viz_url
            }
            
        except Exception as e:
            logger.error(f"Error creating model visualization: {str(e)}")
            raise
    
    def get_visualization_image_path(self, visualization_id):
        """
        Get the file path for a visualization
        
        Args:
            visualization_id: ID of the visualization
            
        Returns:
            Path to the visualization image
        """
        if visualization_id not in self.visualizations:
            return None
        
        try:
            # Get visualization metadata
            viz_metadata = self.visualizations[visualization_id]
            
            # Download from MinIO to temp location
            temp_dir = current_app.config['TEMP_DIR']
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"viz_img_{visualization_id}.png")
            
            self.minio_service.download_file(
                viz_metadata['bucket'], 
                viz_metadata['object_name'], 
                temp_file_path
            )
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error getting visualization image path: {str(e)}")
            return None
    
    def list_visualizations(self, dataset_id=None, model_id=None, viz_type=None):
        """
        List visualizations with optional filtering
        
        Args:
            dataset_id: Optional filter by dataset ID
            model_id: Optional filter by model ID
            viz_type: Optional filter by visualization type
            
        Returns:
            List of visualization metadata
        """
        try:
            visualizations = list(self.visualizations.values())
            
            # Apply filters
            if dataset_id:
                visualizations = [v for v in visualizations if v.get('dataset_id') == dataset_id]
            
            if model_id:
                visualizations = [v for v in visualizations if v.get('model_id') == model_id]
            
            if viz_type:
                visualizations = [v for v in visualizations if v.get('visualization_type') == viz_type]
            
            # Add URLs to visualizations
            for viz in visualizations:
                try:
                    viz['url'] = self.minio_service.get_object_url(
                        viz['bucket'], 
                        viz['object_name']
                    )
                except Exception as e:
                    logger.error(f"Error getting URL for visualization {viz['id']}: {str(e)}")
                    viz['url'] = None
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error listing visualizations: {str(e)}")
            raise
    
    def delete_visualization(self, visualization_id):
        """
        Delete a visualization
        
        Args:
            visualization_id: ID of the visualization
            
        Returns:
            Boolean indicating success
        """
        if visualization_id not in self.visualizations:
            return False
        
        try:
            # Get visualization metadata
            viz_metadata = self.visualizations[visualization_id]
            
            # Delete the object from MinIO
            self.minio_service.delete_file(viz_metadata['bucket'], viz_metadata['object_name'])
            
            # Remove from metadata store
            del self.visualizations[visualization_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting visualization: {str(e)}")
            return False
