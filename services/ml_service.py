import os
import pickle
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import current_app

# Import ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

# Configure logging
logger = logging.getLogger(__name__)

class MLService:
    """Service for machine learning model operations"""
    
    def __init__(self):
        """Initialize the ML service"""
        # Dictionary to store loaded models for prediction
        self.loaded_models = {}
        
        # Model factories for creating different types of models
        self.model_factories = {
            # Regression models
            "linear_regression": lambda params: LinearRegression(**params),
            "random_forest_regressor": lambda params: RandomForestRegressor(**params),
            "xgboost_regressor": lambda params: self._create_xgboost_regressor(params),
            
            # Classification models
            "logistic_regression": lambda params: LogisticRegression(**params),
            "random_forest_classifier": lambda params: RandomForestClassifier(**params),
            "xgboost_classifier": lambda params: self._create_xgboost_classifier(params)
        }
        
        # Metric calculators
        self.regression_metrics = {
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
            "mape": lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
        
        self.classification_metrics = {
            "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def _create_xgboost_regressor(self, params):
        """Create an XGBoost regressor"""
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForestRegressor")
            return RandomForestRegressor(**params)
    
    def _create_xgboost_classifier(self, params):
        """Create an XGBoost classifier"""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForestClassifier")
            return RandomForestClassifier(**params)
    
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
    
    def _prepare_data(self, df, target_column, feature_columns=None):
        """
        Prepare data for modeling
        
        Args:
            df: pandas DataFrame
            target_column: Target column name
            feature_columns: List of feature columns (None to use all except target)
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            # Ensure target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
            else:
                # Verify all specified feature columns exist
                missing_columns = [col for col in feature_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Feature columns not found in dataset: {missing_columns}")
            
            # Extract features and target
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle categorical features (simple approach)
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                # Use label encoding for simplicity
                # In a real system, we might use more sophisticated encoding
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle categorical target if needed
            if y.dtype == 'object' or y.dtype == 'category':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def recommend_models(self, file_path, file_type, target_column, problem_type, evaluation_metric=None):
        """
        Recommend models based on dataset characteristics
        
        Args:
            file_path: Path to the file
            file_type: Type of the file (csv, parquet, json)
            target_column: Target column name
            problem_type: Type of problem (classification/regression)
            evaluation_metric: Preferred evaluation metric
            
        Returns:
            Dict with model recommendations
        """
        try:
            # Read the file
            df = self._read_file(file_path, file_type)
            
            # Get basic dataset statistics
            num_rows, num_cols = df.shape
            
            # Check if target is categorical (for classification problems)
            is_target_categorical = False
            if target_column in df.columns:
                if df[target_column].dtype == 'object' or df[target_column].dtype == 'category':
                    is_target_categorical = True
                else:
                    # Check if it has few unique values
                    unique_values = df[target_column].nunique()
                    if unique_values < 10:  # Arbitrary threshold
                        is_target_categorical = True
            
            # Check if problem type matches target characteristic
            if problem_type == 'classification' and not is_target_categorical:
                logger.warning("Regression target detected but classification problem specified")
            elif problem_type == 'regression' and is_target_categorical:
                logger.warning("Categorical target detected but regression problem specified")
            
            # Determine appropriate models based on dataset size and problem type
            recommendations = []
            
            if problem_type == 'classification':
                # For small datasets
                if num_rows < 10000:
                    recommendations.append({
                        "model": "random_forest_classifier",
                        "reason": "Good performance on small-to-medium datasets, handles non-linearity well",
                        "suggested_hyperparameters": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "min_samples_split": 5
                        }
                    })
                    recommendations.append({
                        "model": "logistic_regression",
                        "reason": "Good baseline model, interpretable, works well with small datasets",
                        "suggested_hyperparameters": {
                            "C": 1.0,
                            "penalty": "l2",
                            "solver": "lbfgs",
                            "max_iter": 1000
                        }
                    })
                
                # For larger datasets
                else:
                    recommendations.append({
                        "model": "xgboost_classifier",
                        "reason": "Excellent performance on large datasets, handles complex relationships",
                        "suggested_hyperparameters": {
                            "n_estimators": 200,
                            "learning_rate": 0.1,
                            "max_depth": 6
                        }
                    })
                    recommendations.append({
                        "model": "random_forest_classifier",
                        "reason": "Good balance of performance and training speed",
                        "suggested_hyperparameters": {
                            "n_estimators": 200,
                            "max_depth": 15,
                            "min_samples_split": 10
                        }
                    })
            
            else:  # regression
                # For small datasets
                if num_rows < 10000:
                    recommendations.append({
                        "model": "random_forest_regressor",
                        "reason": "Handles non-linearity well, resistant to overfitting",
                        "suggested_hyperparameters": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "min_samples_split": 5
                        }
                    })
                    recommendations.append({
                        "model": "linear_regression",
                        "reason": "Simple baseline model, interpretable",
                        "suggested_hyperparameters": {}
                    })
                
                # For larger datasets
                else:
                    recommendations.append({
                        "model": "xgboost_regressor",
                        "reason": "Excellent performance on large datasets, handles complex relationships",
                        "suggested_hyperparameters": {
                            "n_estimators": 200,
                            "learning_rate": 0.1,
                            "max_depth": 6
                        }
                    })
                    recommendations.append({
                        "model": "random_forest_regressor",
                        "reason": "Good balance of performance and training speed",
                        "suggested_hyperparameters": {
                            "n_estimators": 200,
                            "max_depth": 15,
                            "min_samples_split": 10
                        }
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending models: {str(e)}")
            raise
    
    def get_model_hyperparameters(self, model_name):
        """
        Get hyperparameters for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with hyperparameter information
        """
        try:
            if model_name == "linear_regression":
                return {
                    "fit_intercept": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to calculate the intercept for this model"
                    },
                    "normalize": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to normalize the data before fitting"
                    }
                }
            
            elif model_name == "logistic_regression":
                return {
                    "C": {
                        "type": "float",
                        "default": 1.0,
                        "range": [0.01, 100.0],
                        "description": "Inverse of regularization strength"
                    },
                    "penalty": {
                        "type": "categorical",
                        "default": "l2",
                        "options": ["l1", "l2", "elasticnet", "none"],
                        "description": "Penalty norm to use"
                    },
                    "solver": {
                        "type": "categorical",
                        "default": "lbfgs",
                        "options": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "description": "Algorithm to use for optimization"
                    },
                    "max_iter": {
                        "type": "integer",
                        "default": 100,
                        "range": [50, 1000],
                        "description": "Maximum number of iterations"
                    }
                }
            
            elif model_name == "random_forest_regressor" or model_name == "random_forest_classifier":
                return {
                    "n_estimators": {
                        "type": "integer",
                        "default": 100,
                        "range": [10, 1000],
                        "description": "Number of trees in the forest"
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": None,
                        "range": [1, 50],
                        "description": "Maximum depth of the trees"
                    },
                    "min_samples_split": {
                        "type": "integer",
                        "default": 2,
                        "range": [2, 20],
                        "description": "Minimum samples required to split a node"
                    },
                    "min_samples_leaf": {
                        "type": "integer",
                        "default": 1,
                        "range": [1, 20],
                        "description": "Minimum samples required at a leaf node"
                    },
                    "bootstrap": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to use bootstrap samples"
                    }
                }
            
            elif model_name == "xgboost_regressor" or model_name == "xgboost_classifier":
                return {
                    "n_estimators": {
                        "type": "integer",
                        "default": 100,
                        "range": [10, 1000],
                        "description": "Number of boosting rounds"
                    },
                    "learning_rate": {
                        "type": "float",
                        "default": 0.1,
                        "range": [0.01, 1.0],
                        "description": "Step size shrinkage used to prevent overfitting"
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 6,
                        "range": [1, 20],
                        "description": "Maximum depth of a tree"
                    },
                    "subsample": {
                        "type": "float",
                        "default": 1.0,
                        "range": [0.5, 1.0],
                        "description": "Subsample ratio of the training instances"
                    },
                    "colsample_bytree": {
                        "type": "float",
                        "default": 1.0,
                        "range": [0.5, 1.0],
                        "description": "Subsample ratio of columns when constructing each tree"
                    }
                }
            
            else:
                return {"error": f"Model {model_name} not supported"}
                
        except Exception as e:
            logger.error(f"Error getting model hyperparameters: {str(e)}")
            raise
    
    def compare_models(self, file_path, file_type, target_column, problem_type, models, 
                     evaluation_metrics, cv_folds=5):
        """
        Compare multiple models on a dataset
        
        Args:
            file_path: Path to the file
            file_type: Type of the file (csv, parquet, json)
            target_column: Target column name
            problem_type: Type of problem (classification/regression)
            models: List of models to compare
            evaluation_metrics: List of evaluation metrics to use
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dict with comparison results
        """
        try:
            # Read the file
            df = self._read_file(file_path, file_type)
            
            # Prepare data
            X, y, feature_names = self._prepare_data(df, target_column)
            
            # Select appropriate metrics
            if problem_type == 'classification':
                metrics = {m: self.classification_metrics[m] for m in evaluation_metrics 
                          if m in self.classification_metrics}
            else:  # regression
                metrics = {m: self.regression_metrics[m] for m in evaluation_metrics 
                          if m in self.regression_metrics}
            
            if not metrics:
                raise ValueError(f"No valid metrics found for problem type {problem_type}")
            
            # Compare models
            results = []
            
            for model_config in models:
                model_name = model_config['name']
                hyperparameters = model_config.get('hyperparameters', {})
                
                # Check if model is supported
                if model_name not in self.model_factories:
                    logger.warning(f"Model {model_name} not supported, skipping")
                    continue
                
                # Create model
                model = self.model_factories[model_name](hyperparameters)
                
                # Evaluate model with cross-validation
                model_results = {"model": model_name, "hyperparameters": hyperparameters, "metrics": {}}
                
                for metric_name, metric_func in metrics.items():
                    try:
                        # For classification metrics that require class predictions, not probabilities
                        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=self._get_sklearn_scorer(metric_name))
                        model_results["metrics"][metric_name] = {
                            "mean": float(np.mean(cv_scores)),
                            "std": float(np.std(cv_scores)),
                            "values": [float(score) for score in cv_scores]
                        }
                    except Exception as e:
                        logger.error(f"Error calculating {metric_name} for {model_name}: {str(e)}")
                        model_results["metrics"][metric_name] = {"error": str(e)}
                
                results.append(model_results)
            
            # Sort results by first metric's mean value (descending)
            if results and 'metrics' in results[0] and results[0]['metrics']:
                first_metric = list(results[0]['metrics'].keys())[0]
                results.sort(key=lambda x: x['metrics'].get(first_metric, {}).get('mean', 0), reverse=True)
            
            return {
                "results": results,
                "num_samples": len(y),
                "num_features": len(feature_names),
                "cv_folds": cv_folds
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def _get_sklearn_scorer(self, metric_name):
        """Convert our metric name to sklearn scoring parameter"""
        mapping = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        }
        return mapping.get(metric_name)
    
    def train_model(self, file_path, file_type, target_column, feature_columns, model_name,
                  hyperparameters, model_path, test_size=0.2, random_state=42):
        """
        Train a machine learning model
        
        Args:
            file_path: Path to the file
            file_type: Type of the file (csv, parquet, json)
            target_column: Target column name
            feature_columns: List of feature columns (None for all)
            model_name: Name of the model to train
            hyperparameters: Dict of hyperparameters
            model_path: Path where the model should be saved
            test_size: Test set size (0-1)
            random_state: Random seed
            
        Returns:
            Dict with training results
        """
        try:
            # Read the file
            df = self._read_file(file_path, file_type)
            
            # Prepare data
            X, y, feature_names = self._prepare_data(df, target_column, feature_columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Check if model is supported
            if model_name not in self.model_factories:
                raise ValueError(f"Model {model_name} not supported")
            
            # Create and train model
            model = self.model_factories[model_name](hyperparameters)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {}
            
            # Determine if it's a classification or regression problem
            if "classifier" in model_name:
                metric_functions = self.classification_metrics
            else:
                metric_functions = self.regression_metrics
            
            # Calculate all available metrics
            for metric_name, metric_func in metric_functions.items():
                try:
                    metrics[metric_name] = float(metric_func(y_test, y_pred))
                except Exception as e:
                    logger.error(f"Error calculating {metric_name}: {str(e)}")
            
            # Calculate feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in zip(feature_names, model.feature_importances_)
                ]
                # Sort by importance
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            
            # Save the model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Return results
            result = {
                "metrics": metrics,
                "feature_columns": feature_names,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            if feature_importance:
                result["feature_importance"] = feature_importance
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def cross_validate_model(self, file_path, file_type, target_column, feature_columns, model_name,
                           hyperparameters, cv_folds, evaluation_metrics, random_state=42):
        """
        Perform cross-validation for a model
        
        Args:
            file_path: Path to the file
            file_type: Type of the file (csv, parquet, json)
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
            # Read the file
            df = self._read_file(file_path, file_type)
            
            # Prepare data
            X, y, feature_names = self._prepare_data(df, target_column, feature_columns)
            
            # Check if model is supported
            if model_name not in self.model_factories:
                raise ValueError(f"Model {model_name} not supported")
            
            # Create model
            model = self.model_factories[model_name](hyperparameters)
            
            # Determine if it's a classification or regression problem
            if "classifier" in model_name:
                metrics = {m: self.classification_metrics[m] for m in evaluation_metrics 
                          if m in self.classification_metrics}
            else:
                metrics = {m: self.regression_metrics[m] for m in evaluation_metrics 
                          if m in self.regression_metrics}
            
            if not metrics:
                raise ValueError(f"No valid metrics found for model {model_name}")
            
            # Perform cross-validation for each metric
            cv_results = {}
            
            for metric_name, metric_func in metrics.items():
                try:
                    # Use sklearn's cross_val_score with appropriate scorer
                    cv_scores = cross_val_score(
                        model, X, y, cv=cv_folds, 
                        scoring=self._get_sklearn_scorer(metric_name), 
                        n_jobs=-1
                    )
                    
                    cv_results[metric_name] = {
                        "mean": float(np.mean(cv_scores)),
                        "std": float(np.std(cv_scores)),
                        "values": [float(score) for score in cv_scores]
                    }
                except Exception as e:
                    logger.error(f"Error in cross-validation for {metric_name}: {str(e)}")
                    cv_results[metric_name] = {"error": str(e)}
            
            return {
                "model": model_name,
                "hyperparameters": hyperparameters,
                "metrics": cv_results,
                "num_samples": len(y),
                "num_features": len(feature_names),
                "cv_folds": cv_folds
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def load_model(self, deployment_id, model_path):
        """
        Load a model for deployment
        
        Args:
            deployment_id: ID of the deployment
            model_path: Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.loaded_models[deployment_id] = model
            logger.info(f"Model loaded for deployment {deployment_id}")
            
        except Exception as e:
            logger.error(f"Error loading model for deployment {deployment_id}: {str(e)}")
            raise
    
    def unload_model(self, deployment_id):
        """
        Unload a deployed model
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id in self.loaded_models:
            del self.loaded_models[deployment_id]
            logger.info(f"Model unloaded for deployment {deployment_id}")
    
    def predict(self, deployment_id, data):
        """
        Make predictions using a deployed model
        
        Args:
            deployment_id: ID of the deployment
            data: List of input data points
            
        Returns:
            List of predictions
        """
        if deployment_id not in self.loaded_models:
            raise ValueError(f"No model loaded for deployment {deployment_id}")
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Get model
            model = self.loaded_models[deployment_id]
            
            # Handle categorical features (simple approach)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                # Use label encoding for simplicity
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            
            # Make predictions
            predictions = model.predict(df)
            
            # Convert to list
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Error making predictions for deployment {deployment_id}: {str(e)}")
            raise
    
    def batch_predict(self, deployment_id, input_path, output_path, file_type, feature_columns=None):
        """
        Make batch predictions using a deployed model
        
        Args:
            deployment_id: ID of the deployment
            input_path: Path to the input file
            output_path: Path where the output should be saved
            file_type: Type of the file (csv, parquet, json)
            feature_columns: List of feature columns to use (None for all)
            
        Returns:
            Dict with batch prediction results
        """
        if deployment_id not in self.loaded_models:
            raise ValueError(f"No model loaded for deployment {deployment_id}")
        
        try:
            # Read the input file
            df = self._read_file(input_path, file_type)
            
            # Extract features
            if feature_columns:
                X = df[feature_columns]
            else:
                X = df
            
            # Handle categorical features (simple approach)
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                # Use label encoding for simplicity
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Get model
            model = self.loaded_models[deployment_id]
            
            # Make predictions
            predictions = model.predict(X)
            
            # Add predictions to the original DataFrame
            df['prediction'] = predictions
            
            # Write to output file
            if file_type == 'csv':
                df.to_csv(output_path, index=False)
            elif file_type == 'parquet':
                df.to_parquet(output_path, index=False)
            elif file_type == 'json' or file_type == 'jsonl':
                df.to_json(output_path, orient='records')
            
            # Return batch prediction summary
            return {
                "rows_processed": len(df),
                "prediction_column_added": "prediction"
            }
            
        except Exception as e:
            logger.error(f"Error making batch predictions for deployment {deployment_id}: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_importance, output_path):
        """
        Generate a feature importance plot
        
        Args:
            feature_importance: List of feature importance values
            output_path: Path where the plot should be saved
        """
        try:
            # Import matplotlib
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Sort feature importance by importance value
            sorted_features = sorted(feature_importance, key=lambda x: x['importance'])
            
            # Extract feature names and importance values
            features = [item['feature'] for item in sorted_features]
            importances = [item['importance'] for item in sorted_features]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(features, importances)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {str(e)}")
            raise
