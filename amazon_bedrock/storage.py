import os
import boto3
from src.config import Config

class StorageService:
    def __init__(self):
        self._config = Config()
        self._s3_resource = boto3.resource('s3',
                                           aws_access_key_id=self._config.access_key,
                                           aws_secret_access_key=self._config.secret_Key,
                                           region_name=self._config.region)
        self._s3 = self._s3_resource.Bucket(self._config.s3_bucket_name)

    def download_file(self, file_name: str, bucket_dir: str) -> str:
        key = os.path.join(bucket_dir, file_name)
        cache_file_path = os.path.join(self._config.s3_cache_dir, key)
        try:
            self._s3.download_file(key, cache_file_path)
            return cache_file_path
        except Exception as e:
            print(f"Error downloading file {key}: {e}")
            raise

    def download_all_files(self, bucket_dir: str):
        result = []
        for object_summary in self._s3.objects.filter(Prefix=bucket_dir):
            if not object_summary.key.endswith('/'):
                file_name = os.path.basename(object_summary.key)
                cache_file_path = os.path.join(self._config.s3_cache_dir, bucket_dir, file_name)
                result.append((file_name, cache_file_path))
                self.download_file(file_name, bucket_dir)
        return result
