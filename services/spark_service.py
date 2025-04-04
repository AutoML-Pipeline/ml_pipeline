import os
import logging
import json
import csv
import tempfile
from datetime import datetime
from flask import current_app

# Configure logging
logger = logging.getLogger(__name__)

class SparkService:
    """Service for interacting with Apache Spark for data processing"""
    
    def __init__(self):
        """Initialize the Spark service"""
        try:
            from pyspark.sql import SparkSession
            from pyspark import SparkConf
            
            # Get configuration from app config
            app_name = current_app.config.get('SPARK_APP_NAME', 'ML-Pipeline')
            master = current_app.config.get('SPARK_MASTER', 'local[*]')
            executor_memory = current_app.config.get('SPARK_EXECUTOR_MEMORY', '2g')
            driver_memory = current_app.config.get('SPARK_DRIVER_MEMORY', '2g')
            
            # Create Spark configuration
            conf = SparkConf().setAppName(app_name).setMaster(master)
            conf.set("spark.executor.memory", executor_memory)
            conf.set("spark.driver.memory", driver_memory)
            
            # Create Spark session
            self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
            logger.info(f"Initialized SparkService with app_name={app_name}, master={master}")
        except Exception as e:
            logger.error(f"Error initializing SparkService: {str(e)}")
            raise
    
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
            # Read file based on type
            if file_type == 'csv':
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
            elif file_type == 'parquet':
                df = self.spark.read.parquet(file_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df = self.spark.read.json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Get schema
            schema = []
            for field in df.schema.fields:
                schema.append({
                    "name": field.name,
                    "type": str(field.dataType)
                })
            
            # Get preview data
            preview_rows = df.limit(limit).collect()
            rows = []
            for row in preview_rows:
                rows.append(row.asDict())
                
            preview_data = {
                "num_rows": df.count(),
                "num_columns": len(df.columns),
                "rows": rows
            }
            
            return preview_data, schema
            
        except Exception as e:
            logger.error(f"Error in get_data_preview_and_schema: {str(e)}")
            # If Spark fails, fall back to a simpler method
            return self._fallback_preview_and_schema(file_path, file_type, limit)
    
    def _fallback_preview_and_schema(self, file_path, file_type, limit=10):
        """Fallback method for preview when Spark fails"""
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
                        if i == 0:
                            # Try to infer types from first row
                            for j, val in enumerate(row):
                                if j < len(headers):
                                    try:
                                        int(val)
                                        schema[j]["type"] = "integer"
                                    except ValueError:
                                        try:
                                            float(val)
                                            schema[j]["type"] = "double"
                                        except ValueError:
                                            schema[j]["type"] = "string"
                        
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
                # For other file types just return empty data
                preview_data = {
                    "num_rows": 0,
                    "num_columns": 0,
                    "rows": []
                }
                schema = []
            
            return preview_data, schema
            
        except Exception as e:
            logger.error(f"Error in fallback_preview_and_schema: {str(e)}")
            # Return minimal data
            return {
                "num_rows": 0,
                "num_columns": 0,
                "rows": []
            }, []
    
    def preprocess_data(self, input_path, output_path, operations, file_type):
        """
        Perform preprocessing operations on a dataset
        
        Args:
            input_path: Path to the input file
            output_path: Path to save the output file
            operations: List of preprocessing operations
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Dict with processing results
        """
        try:
            # Read data
            if file_type == 'csv':
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
            elif file_type == 'parquet':
                df = self.spark.read.parquet(input_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df = self.spark.read.json(input_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Apply operations
            for op in operations:
                op_type = op.get('type')
                
                if op_type == 'remove_nulls':
                    columns = op.get('columns', [])
                    if columns:
                        df = df.dropna(subset=columns)
                    else:
                        df = df.dropna()
                
                elif op_type == 'fillna':
                    columns = op.get('columns', [])
                    value = op.get('value', 0)
                    
                    if columns:
                        for column in columns:
                            df = df.fillna(value, subset=[column])
                    else:
                        df = df.fillna(value)
                
                elif op_type == 'normalize':
                    from pyspark.ml.feature import MinMaxScaler, StandardScaler
                    from pyspark.ml.feature import VectorAssembler
                    from pyspark.sql.functions import col
                    
                    columns = op.get('columns', [])
                    method = op.get('method', 'minmax')
                    
                    if columns:
                        # Create vector assembler
                        assembler = VectorAssembler(inputCols=columns, outputCol="features")
                        assembled_df = assembler.transform(df)
                        
                        # Apply scaling
                        if method == 'minmax':
                            scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
                        else:  # zscore
                            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
                        
                        # Fit and transform
                        scaler_model = scaler.fit(assembled_df)
                        scaled_df = scaler_model.transform(assembled_df)
                        
                        # TODO: Extract individual columns from the vector
                        # This is a simplified implementation
                        df = scaled_df
                
                elif op_type == 'outlier_removal':
                    from pyspark.sql.functions import col
                    
                    columns = op.get('columns', [])
                    method = op.get('method', 'iqr')
                    threshold = op.get('threshold', 1.5)
                    
                    if columns and method == 'iqr':
                        for column in columns:
                            # Calculate quantiles
                            quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
                            q1, q3 = quantiles[0], quantiles[1]
                            iqr = q3 - q1
                            
                            # Filter outliers
                            lower_bound = q1 - threshold * iqr
                            upper_bound = q3 + threshold * iqr
                            
                            df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
                    
                    elif columns and method == 'zscore':
                        # Calculate mean and std for each column
                        for column in columns:
                            from pyspark.sql.functions import mean, stddev
                            
                            stats = df.select(mean(col(column)).alias('mean'), 
                                             stddev(col(column)).alias('stddev')).collect()[0]
                            
                            mean_val = stats['mean']
                            stddev_val = stats['stddev']
                            
                            # Filter based on z-score
                            df = df.filter(abs((col(column) - mean_val) / stddev_val) <= threshold)
            
            # Write output file
            if file_type == 'csv':
                df.write.option("header", "true").mode("overwrite").csv(output_path + "_temp")
                # Spark writes to a directory with part files, so combine them
                self._combine_part_files(output_path + "_temp", output_path)
            elif file_type == 'parquet':
                df.write.mode("overwrite").parquet(output_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df.write.mode("overwrite").json(output_path + "_temp")
                # Spark writes to a directory with part files, so combine them
                self._combine_part_files(output_path + "_temp", output_path)
            
            return {
                "status": "success",
                "operations_applied": len(operations),
                "rows_processed": df.count(),
                "columns_processed": len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            # If Spark fails, do a simple file copy as fallback
            import shutil
            shutil.copy(input_path, output_path)
            return {
                "status": "error",
                "message": str(e),
                "fallback": "Used file copy as fallback"
            }
    
    def _combine_part_files(self, input_dir, output_file):
        """Combine Spark part files into a single output file"""
        import glob
        
        with open(output_file, 'w') as outfile:
            # For CSV, write header from first part file, then data from all
            part_files = glob.glob(os.path.join(input_dir, "part-*"))
            if part_files:
                with open(part_files[0], 'r') as first_part:
                    outfile.write(first_part.read())
                
                for part_file in part_files[1:]:
                    with open(part_file, 'r') as part:
                        # Skip header line for subsequent files
                        next(part, None)
                        outfile.write(part.read())
    
    def engineer_features(self, input_path, output_path, operations, file_type):
        """
        Perform feature engineering operations
        
        Args:
            input_path: Path to the input file
            output_path: Path to save the output file
            operations: List of feature engineering operations
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Dict with processing results
        """
        try:
            # Read data
            if file_type == 'csv':
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
            elif file_type == 'parquet':
                df = self.spark.read.parquet(input_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df = self.spark.read.json(input_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Apply operations
            features_created = 0
            
            for op in operations:
                op_type = op.get('type')
                
                if op_type == 'polynomial_features':
                    from pyspark.ml.feature import PolynomialExpansion
                    from pyspark.ml.feature import VectorAssembler
                    from pyspark.sql.functions import col
                    
                    columns = op.get('columns', [])
                    degree = op.get('degree', 2)
                    
                    if columns:
                        # Create vector assembler
                        assembler = VectorAssembler(inputCols=columns, outputCol="features")
                        assembled_df = assembler.transform(df)
                        
                        # Apply polynomial expansion
                        poly = PolynomialExpansion(degree=degree, inputCol="features", outputCol="poly_features")
                        df = poly.transform(assembled_df)
                        
                        features_created += len(columns) ** degree - len(columns)
                
                elif op_type == 'binning':
                    from pyspark.ml.feature import Bucketizer
                    from pyspark.sql.functions import col, min, max
                    
                    column = op.get('column')
                    bins = op.get('bins', 5)
                    
                    if column:
                        # Get min/max values
                        stats = df.select(min(col(column)).alias('min'), max(col(column)).alias('max')).collect()[0]
                        min_val = stats['min']
                        max_val = stats['max']
                        
                        # Create splits
                        step = (max_val - min_val) / bins
                        splits = [min_val + i * step for i in range(bins + 1)]
                        splits[0] = float('-inf')
                        splits[-1] = float('inf')
                        
                        # Apply binning
                        bucketizer = Bucketizer(splits=splits, inputCol=column, outputCol=f"{column}_binned")
                        df = bucketizer.transform(df)
                        
                        features_created += 1
                
                elif op_type == 'text_vectorization':
                    from pyspark.ml.feature import CountVectorizer, HashingTF, IDF
                    from pyspark.ml.feature import Tokenizer, StopWordsRemover
                    
                    column = op.get('column')
                    method = op.get('method', 'tfidf')
                    max_features = op.get('max_features', 1000)
                    
                    if column:
                        # Tokenize
                        tokenizer = Tokenizer(inputCol=column, outputCol=f"{column}_tokens")
                        wordsData = tokenizer.transform(df)
                        
                        # Remove stop words
                        remover = StopWordsRemover(inputCol=f"{column}_tokens", outputCol=f"{column}_filtered")
                        wordsData = remover.transform(wordsData)
                        
                        if method == 'count':
                            # Apply count vectorization
                            cv = CountVectorizer(inputCol=f"{column}_filtered", outputCol=f"{column}_vector", 
                                              maxDF=0.8, minDF=3.0, vocabSize=max_features)
                            cv_model = cv.fit(wordsData)
                            df = cv_model.transform(wordsData)
                        else:  # tfidf
                            # Apply TF-IDF
                            hashingTF = HashingTF(inputCol=f"{column}_filtered", outputCol=f"{column}_tf", 
                                               numFeatures=max_features)
                            featurized_data = hashingTF.transform(wordsData)
                            
                            idf = IDF(inputCol=f"{column}_tf", outputCol=f"{column}_vector")
                            idf_model = idf.fit(featurized_data)
                            df = idf_model.transform(featurized_data)
                        
                        features_created += max_features
                
                elif op_type == 'pca':
                    from pyspark.ml.feature import PCA
                    from pyspark.ml.feature import VectorAssembler
                    
                    columns = op.get('columns', [])
                    n_components = op.get('n_components', 2)
                    
                    if columns and n_components < len(columns):
                        # Create vector assembler
                        assembler = VectorAssembler(inputCols=columns, outputCol="features")
                        assembled_df = assembler.transform(df)
                        
                        # Apply PCA
                        pca = PCA(k=n_components, inputCol="features", outputCol="pca_features")
                        pca_model = pca.fit(assembled_df)
                        df = pca_model.transform(assembled_df)
                        
                        features_created += n_components
            
            # Write output file
            if file_type == 'csv':
                df.write.option("header", "true").mode("overwrite").csv(output_path + "_temp")
                # Spark writes to a directory with part files, so combine them
                self._combine_part_files(output_path + "_temp", output_path)
            elif file_type == 'parquet':
                df.write.mode("overwrite").parquet(output_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df.write.mode("overwrite").json(output_path + "_temp")
                # Spark writes to a directory with part files, so combine them
                self._combine_part_files(output_path + "_temp", output_path)
            
            return {
                "status": "success",
                "operations_applied": len(operations),
                "features_created": features_created,
                "rows_processed": df.count(),
                "columns_processed": len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error in engineer_features: {str(e)}")
            # If Spark fails, do a simple file copy as fallback
            import shutil
            shutil.copy(input_path, output_path)
            return {
                "status": "error",
                "message": str(e),
                "fallback": "Used file copy as fallback"
            }
    
    def calculate_feature_importance(self, input_path, target_column, method, file_type):
        """
        Calculate feature importance
        
        Args:
            input_path: Path to the input file
            target_column: Target column name
            method: Method to use (mutual_info, chi2, etc.)
            file_type: Type of file (csv, parquet, json)
            
        Returns:
            Dict with feature importance scores
        """
        try:
            # Read data
            if file_type == 'csv':
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
            elif file_type == 'parquet':
                df = self.spark.read.parquet(input_path)
            elif file_type == 'json' or file_type == 'jsonl':
                df = self.spark.read.json(input_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Get feature columns (all columns except target)
            feature_columns = [c for c in df.columns if c != target_column]
            
            # Calculate feature importance
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.classification import RandomForestClassifier
            from pyspark.ml.regression import RandomForestRegressor
            from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
            
            # Create vector assembler
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            assembled_df = assembler.transform(df)
            
            # Train a random forest to get feature importance
            # Detect if classification or regression
            target_type = df.schema[target_column].dataType.simpleString()
            is_classification = target_type in ['string', 'boolean'] or 'int' in target_type
            
            if is_classification:
                model = RandomForestClassifier(labelCol=target_column, featuresCol="features", numTrees=10)
                model_fit = model.fit(assembled_df)
                importance_values = model_fit.featureImportances.toArray()
            else:
                model = RandomForestRegressor(labelCol=target_column, featuresCol="features", numTrees=10)
                model_fit = model.fit(assembled_df)
                importance_values = model_fit.featureImportances.toArray()
            
            # Create result dictionary
            feature_importance = {}
            for i, col in enumerate(feature_columns):
                if i < len(importance_values):
                    feature_importance[col] = float(importance_values[i])
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "method": method,
                "target_column": target_column,
                "features": [{"name": f[0], "importance": f[1]} for f in sorted_features]
            }
            
        except Exception as e:
            logger.error(f"Error in calculate_feature_importance: {str(e)}")
            # If Spark fails, return mock data
            importance_dict = {}
            for col in df.columns:
                if col != target_column:
                    # Generate a random importance value
                    importance_dict[col] = 1.0 / len(df.columns)
            
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "method": method,
                "target_column": target_column,
                "features": [{"name": f[0], "importance": f[1]} for f in sorted_features],
                "error": str(e),
                "fallback": "Used random values as fallback"
            }