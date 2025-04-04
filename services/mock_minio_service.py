import os
import logging
import json
import tempfile
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MockMinioService:
    """Mock implementation of MinIO service for development and testing"""
    
    def __init__(self):
        """Initialize the mock storage system"""
        self.storage_dir = tempfile.mkdtemp(prefix="mock_minio_")
        self.buckets = {}
        logger.info(f"Initialized MockMinioService with storage directory: {self.storage_dir}")
    
    def ensure_bucket_exists(self, bucket_name):
        """Ensure that the specified bucket exists, create if not"""
        if bucket_name not in self.buckets:
            bucket_dir = os.path.join(self.storage_dir, bucket_name)
            os.makedirs(bucket_dir, exist_ok=True)
            self.buckets[bucket_name] = bucket_dir
            logger.info(f"Created mock bucket: {bucket_name} at {bucket_dir}")
        return True
    
    def upload_file(self, bucket_name, object_name, file_path):
        """
        Upload a file to mock storage
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name to assign to the object
            file_path: Path to the file to upload
        """
        self.ensure_bucket_exists(bucket_name)
        
        # Create destination path
        dest_path = os.path.join(self.storage_dir, bucket_name, object_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy file
        with open(file_path, 'rb') as src_file, open(dest_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        file_size = os.path.getsize(dest_path)
        logger.info(f"Uploaded {file_path} ({file_size} bytes) to mock {bucket_name}/{object_name}")
        
        # Create metadata file
        metadata_path = dest_path + ".metadata"
        metadata = {
            "size": file_size,
            "content_type": "application/octet-stream",
            "last_modified": datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file)
        
        return True
    
    def download_file(self, bucket_name, object_name, file_path):
        """
        Download a file from mock storage
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            file_path: Path where the file should be saved
        """
        source_path = os.path.join(self.storage_dir, bucket_name, object_name)
        
        if not os.path.exists(source_path):
            logger.error(f"Object does not exist: {bucket_name}/{object_name}")
            raise FileNotFoundError(f"Object {bucket_name}/{object_name} does not exist")
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Copy file
        with open(source_path, 'rb') as src_file, open(file_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        logger.info(f"Downloaded mock {bucket_name}/{object_name} to {file_path}")
        return True
    
    def delete_file(self, bucket_name, object_name):
        """
        Delete a file from mock storage
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
        """
        file_path = os.path.join(self.storage_dir, bucket_name, object_name)
        metadata_path = file_path + ".metadata"
        
        if not os.path.exists(file_path):
            logger.error(f"Object does not exist: {bucket_name}/{object_name}")
            raise FileNotFoundError(f"Object {bucket_name}/{object_name} does not exist")
        
        # Remove file and metadata
        os.remove(file_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        logger.info(f"Deleted mock {bucket_name}/{object_name}")
        return True
    
    def list_objects(self, bucket_name, prefix='', recursive=True):
        """
        List objects in a mock bucket
        
        Args:
            bucket_name: Name of the bucket
            prefix: Filter by prefix
            recursive: If True, list objects recursively
            
        Returns:
            List of object information
        """
        self.ensure_bucket_exists(bucket_name)
        bucket_path = os.path.join(self.storage_dir, bucket_name)
        
        result = []
        for root, _, files in os.walk(bucket_path):
            for filename in files:
                if filename.endswith(".metadata"):
                    continue
                
                rel_path = os.path.relpath(os.path.join(root, filename), bucket_path)
                
                # Apply prefix filter
                if prefix and not rel_path.startswith(prefix):
                    continue
                
                # Apply recursive filter
                if not recursive and os.path.sep in rel_path:
                    continue
                
                # Get metadata
                file_path = os.path.join(root, filename)
                metadata_path = file_path + ".metadata"
                
                size = os.path.getsize(file_path)
                last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as metadata_file:
                            metadata = json.load(metadata_file)
                            if 'size' in metadata:
                                size = metadata['size']
                            if 'last_modified' in metadata:
                                last_modified = datetime.fromisoformat(metadata['last_modified'])
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {rel_path}: {str(e)}")
                
                result.append({
                    'name': rel_path,
                    'size': size,
                    'last_modified': last_modified
                })
        
        return result
    
    def get_object_url(self, bucket_name, object_name, expires=3600):
        """
        Get a mock URL for an object
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            expires: Expiration time in seconds
            
        Returns:
            Mock URL
        """
        # For a mock service, just return a fake URL
        return f"file://{self.storage_dir}/{bucket_name}/{object_name}?expires={expires}"