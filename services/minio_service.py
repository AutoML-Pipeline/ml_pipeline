import os
import logging
import tempfile
from minio import Minio
from minio.error import S3Error
from flask import current_app

# Configure logging
logger = logging.getLogger(__name__)

class MinioService:
    """Service for interacting with MinIO object storage"""
    
    def __init__(self, endpoint=None, access_key=None, secret_key=None, secure=None):
        """Initialize the MinIO client with provided parameters or defaults"""
        self.client = None
        
        # Use Flask app config if available, otherwise use parameters or defaults
        try:
            from flask import current_app
            self.endpoint = endpoint if endpoint is not None else current_app.config.get('MINIO_ENDPOINT', '127.0.0.1:9090')
            self.access_key = access_key if access_key is not None else current_app.config.get('MINIO_ACCESS_KEY', '')
            self.secret_key = secret_key if secret_key is not None else current_app.config.get('MINIO_SECRET_KEY', '')
            self.secure = secure if secure is not None else current_app.config.get('MINIO_SECURE', False)
        except RuntimeError:
            # We're outside of application context
            self.endpoint = endpoint if endpoint is not None else '127.0.0.1:9090'
            self.access_key = access_key if access_key is not None else ''
            self.secret_key = secret_key if secret_key is not None else ''
            self.secure = secure if secure is not None else False
            
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize or re-initialize the MinIO client"""
        try:
            # Use instance variables rather than Flask app config
            
            self.client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            
            logger.info(f"MinIO client initialized with endpoint {self.endpoint}")
            
        except Exception as e:
            logger.error(f"Error initializing MinIO client: {str(e)}")
            raise
    
    def ensure_bucket_exists(self, bucket_name):
        """Ensure that the specified bucket exists, create if not"""
        try:
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"Creating bucket {bucket_name}")
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket {bucket_name} created successfully")
            return True
        except S3Error as e:
            logger.error(f"Error ensuring bucket {bucket_name} exists: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error ensuring bucket {bucket_name} exists: {str(e)}")
            raise
    
    def upload_file(self, bucket_name, object_name, file_path):
        """
        Upload a file to MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name to assign to the object
            file_path: Path to the file to upload
        """
        try:
            # Ensure bucket exists
            self.ensure_bucket_exists(bucket_name)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Upload the file
            logger.info(f"Uploading file {file_path} to {bucket_name}/{object_name}")
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type='application/octet-stream'
            )
            
            logger.info(f"Uploaded {file_path} ({file_size} bytes) to {bucket_name}/{object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"S3 error uploading file {file_path} to {bucket_name}/{object_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file {file_path} to {bucket_name}/{object_name}: {str(e)}")
            raise
    
    def download_file(self, bucket_name, object_name, file_path):
        """
        Download a file from MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            file_path: Path where the file should be saved
        """
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Download the file
            logger.info(f"Downloading {bucket_name}/{object_name} to {file_path}")
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path
            )
            
            logger.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")
            return True
            
        except S3Error as e:
            logger.error(f"S3 error downloading {bucket_name}/{object_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading {bucket_name}/{object_name}: {str(e)}")
            raise
    
    def delete_file(self, bucket_name, object_name):
        """
        Delete a file from MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
        """
        try:
            # Delete the object
            logger.info(f"Deleting {bucket_name}/{object_name}")
            self.client.remove_object(
                bucket_name=bucket_name,
                object_name=object_name
            )
            
            logger.info(f"Deleted {bucket_name}/{object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"S3 error deleting {bucket_name}/{object_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting {bucket_name}/{object_name}: {str(e)}")
            raise
    
    def list_objects(self, bucket_name, prefix='', recursive=True):
        """
        List objects in a bucket
        
        Args:
            bucket_name: Name of the bucket
            prefix: Filter by prefix
            recursive: If True, list objects recursively
            
        Returns:
            List of object information
        """
        try:
            # Ensure bucket exists
            self.ensure_bucket_exists(bucket_name)
            
            # List objects
            objects = self.client.list_objects(
                bucket_name=bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            # Convert to list of dictionaries
            result = []
            for obj in objects:
                result.append({
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified
                })
            
            return result
            
        except S3Error as e:
            logger.error(f"S3 error listing objects in {bucket_name}/{prefix}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing objects in {bucket_name}/{prefix}: {str(e)}")
            raise
    
    def get_object_url(self, bucket_name, object_name, expires=3600):
        """
        Get a presigned URL for an object
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            expires: Expiration time in seconds
            
        Returns:
            Presigned URL
        """
        try:
            # Get presigned URL
            url = self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=expires
            )
            
            return url
            
        except S3Error as e:
            logger.error(f"S3 error getting URL for {bucket_name}/{object_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting URL for {bucket_name}/{object_name}: {str(e)}")
            raise
