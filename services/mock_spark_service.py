import os
import logging
import json
import csv
import random
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MockSparkService:
    """Mock implementation of Spark service for development and testing"""
    
    def __init__(self):
        """Initialize the mock Spark service"""
        logger.info("Initialized MockSparkService")
    
    def get_data_preview_and_schema(self, file_path, file_type, limit=10):
        """
        Get a preview of the data and schema
        
        Args:
            file_path: Path to the input file
            file_type: Type of file (csv, parquet, json)
            limit: Maximum number of rows to include in preview
            
        Returns:
            Tuple of (preview_data, schema)
        """
        try:
            preview_data = {
                "num_rows": 0, 
                "num_columns": 0,
                "rows": []
            }
            schema = []
            
            if file_type == 'csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                    
                    # Create schema
                    schema = [{"name": col, "type": "string"} for col in headers]
                    preview_data["num_columns"] = len(headers)
                    
                    # Read rows for preview
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= limit:
                            break
                        row_dict = {headers[j]: val for j, val in enumerate(row) if j < len(headers)}
                        rows.append(row_dict)
                        preview_data["num_rows"] += 1
                    
                    preview_data["rows"] = rows
                    
                    # Estimate total number of rows by file size
                    file_size = os.path.getsize(file_path)
                    if len(rows) > 0:
                        avg_row_size = file_size / (len(rows) + 1)  # +1 for header
                        preview_data["num_rows"] = int(file_size / avg_row_size)
            
            elif file_type == 'json' or file_type == 'jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Try to parse as single JSON object
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            # JSON array
                            if len(data) > 0:
                                sample = data[0]
                                schema = [{"name": k, "type": type(v).__name__} for k, v in sample.items()]
                                preview_data["num_columns"] = len(schema)
                                preview_data["rows"] = data[:limit]
                                preview_data["num_rows"] = len(data)
                        else:
                            # Single JSON object
                            schema = [{"name": k, "type": type(v).__name__} for k, v in data.items()]
                            preview_data["num_columns"] = len(schema)
                            preview_data["rows"] = [data]
                            preview_data["num_rows"] = 1
                    except json.JSONDecodeError:
                        # Try to parse as JSONL
                        f.seek(0)
                        rows = []
                        line_sample = None
                        for i, line in enumerate(f):
                            if i >= limit:
                                break
                            try:
                                line_data = json.loads(line.strip())
                                if not line_sample and isinstance(line_data, dict):
                                    line_sample = line_data
                                rows.append(line_data)
                            except json.JSONDecodeError:
                                pass
                        
                        if line_sample:
                            schema = [{"name": k, "type": type(v).__name__} for k, v in line_sample.items()]
                        
                        preview_data["num_columns"] = len(schema)
                        preview_data["rows"] = rows
                        
                        # Estimate total lines
                        file_size = os.path.getsize(file_path)
                        if len(rows) > 0:
                            avg_row_size = file_size / len(rows)
                            preview_data["num_rows"] = int(file_size / avg_row_size)
            
            else:
                # For other file types just return mock data
                preview_data = {
                    "num_rows": 100,
                    "num_columns": 5,
                    "rows": [
                        {"col1": f"value{i}_1", "col2": i * 2, "col3": i * 3, "col4": f"category_{i % 3}", "col5": f"2023-01-{i % 30 + 1}"}
                        for i in range(limit)
                    ]
                }
                schema = [
                    {"name": "col1", "type": "string"},
                    {"name": "col2", "type": "integer"},
                    {"name": "col3", "type": "integer"},
                    {"name": "col4", "type": "string"},
                    {"name": "col5", "type": "date"}
                ]
            
            return preview_data, schema
            
        except Exception as e:
            logger.error(f"Error in get_data_preview_and_schema: {str(e)}")
            # Return mock data as fallback
            return {
                "num_rows": 100,
                "num_columns": 5,
                "rows": [
                    {"col1": f"value{i}_1", "col2": i * 2, "col3": i * 3, "col4": f"category_{i % 3}", "col5": f"2023-01-{i % 30 + 1}"}
                    for i in range(limit)
                ]
            }, [
                {"name": "col1", "type": "string"},
                {"name": "col2", "type": "integer"},
                {"name": "col3", "type": "integer"},
                {"name": "col4", "type": "string"},
                {"name": "col5", "type": "date"}
            ]
    
    def preprocess_data(self, input_path, output_path, operations, file_type):
        """
        Mock preprocessing on a dataset
        
        Args:
            input_path: Path to the input file
            output_path: Path to save the output file
            operations: List of preprocessing operations
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Dict with processing results
        """
        # For mock service, just copy the file
        with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
            dst.write(src.read())
        
        # Log the operations
        for op in operations:
            logger.info(f"Mock processing operation: {op}")
        
        return {
            "status": "success",
            "operations_applied": len(operations),
            "rows_processed": 1000
        }
    
    def engineer_features(self, input_path, output_path, operations, file_type):
        """
        Mock feature engineering on a dataset
        
        Args:
            input_path: Path to the input file
            output_path: Path to save the output file
            operations: List of feature engineering operations
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Dict with processing results
        """
        # For mock service, just copy the file
        with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
            dst.write(src.read())
        
        # Log the operations
        for op in operations:
            logger.info(f"Mock feature engineering operation: {op}")
        
        return {
            "status": "success",
            "operations_applied": len(operations),
            "features_created": len(operations) * 2
        }
    
    def calculate_feature_importance(self, input_path, target_column, method, file_type):
        """
        Mock feature importance calculation
        
        Args:
            input_path: Path to the input file
            target_column: Target column name
            method: Method to use (mutual_info, chi2, etc.)
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Dict with feature importance scores
        """
        # Get file schema to get column names
        _, schema = self.get_data_preview_and_schema(input_path, file_type)
        
        # Create mock importance scores
        feature_names = [f["name"] for f in schema if f["name"] != target_column]
        importance_scores = {}
        
        for feature in feature_names:
            # Generate a random importance score between 0 and 1
            importance_scores[feature] = round(random.random(), 4)
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "method": method,
            "target_column": target_column,
            "features": [{"name": f[0], "importance": f[1]} for f in sorted_features]
        }