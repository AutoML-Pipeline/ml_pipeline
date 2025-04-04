import os

# Flask configuration
DEBUG = True
SECRET_KEY = os.environ.get("SESSION_SECRET", "default-dev-key")

# MinIO configuration
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "False").lower() == "true"

# MinIO bucket configuration
BUCKET_DATA_RAW = "raw-data"
BUCKET_DATA_PROCESSED = "processed-data"
BUCKET_MODELS = "ml-models"
BUCKET_VISUALIZATIONS = "visualizations"
BUCKET_EVALUATIONS = "model-evaluations"

# Spark configuration
SPARK_APP_NAME = "ML-Pipeline"
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "2g")
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "2g")

# File upload settings
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB limit for uploads
ALLOWED_EXTENSIONS = {'csv', 'parquet', 'json', 'jsonl'}

# ML model settings
AVAILABLE_MODELS = {
    "regression": ["linear_regression", "random_forest_regressor", "xgboost_regressor"],
    "classification": ["logistic_regression", "random_forest_classifier", "xgboost_classifier"]
}

# Feature engineering settings
DEFAULT_FEATURES = {
    "numeric": ["mean", "median", "min", "max", "std"],
    "categorical": ["one_hot_encoding", "label_encoding", "target_encoding"]
}

# Temporary storage paths
TEMP_DIR = "/tmp/ml-pipeline"
