import os

# MinIO Configuration
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = os.environ.get('MINIO_SECURE', 'False').lower() == 'true'

# Buckets
DATA_BUCKET = 'ml-datasets'
MODEL_BUCKET = 'ml-models'
VISUALIZATION_BUCKET = 'ml-visualizations'
EVALUATION_BUCKET = 'ml-evaluations'

# Spark Configuration
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'local[*]')
SPARK_APP_NAME = 'MLPipeline'

# Model Configuration
AVAILABLE_MODELS = [
    'RandomForest',
    'XGBoost',
    'LogisticRegression',
    'DecisionTree',
    'SVM'
]

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Visualization Types
VISUALIZATION_TYPES = [
    'histogram',
    'scatter',
    'correlation',
    'feature_importance',
    'confusion_matrix',
    'roc_curve'
]

# Temporary directory for file operations
TEMP_DIR = '/tmp/ml-pipeline'

# Create temp directory if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
