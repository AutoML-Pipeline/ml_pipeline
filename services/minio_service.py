import os
import logging
import io
from minio import Minio
from minio.error import S3Error
from flask import current_app
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

class MinioService:
    """Service for interacting with MinIO object storage"""
    
    def __init__(self):
        """Initialize the MinIO client"""
        try:
            # Get configuration from Flask app
            app_config = current_app.config
            self.endpoint = app_config.get('MINIO_ENDPOINT', '127.0.0.1:50877')
            self.access_key = app_config.get('MINIO_ACCESS_KEY', 'minioadmin')
            self.secret_key = app_config.get('MINIO_SECRET_KEY', 'minioadmin')
            self.secure = app_config.get('MINIO_SECURE', False)
            
            # Initialize MinIO client
            self.client = Minio(
                self.endpoint,
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
    
    def upload_file(self, bucket_name, object_name, file_data):
        """
        Upload a file to MinIO
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name to assign to the object
            file_data: File data as bytes
        """
        try:
            # Ensure bucket exists
            self.ensure_bucket_exists(bucket_name)
            
            # Create file stream
            file_stream = io.BytesIO(file_data)
            
            # Upload the file
            logger.info(f"Uploading file to {bucket_name}/{object_name}")
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file_stream,
                length=len(file_data)
            )
            
            logger.info(f"Uploaded {object_name} ({len(file_data)} bytes) to {bucket_name}")
            return True
            
        except S3Error as e:
            logger.error(f"S3 error uploading file to {bucket_name}/{object_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file to {bucket_name}/{object_name}: {str(e)}")
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
