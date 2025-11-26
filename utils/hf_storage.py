"""
Hugging Face Hub Storage Manager
Handles upload/download of ChromaDB vector database to/from Hugging Face Hub
"""

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import os
import shutil
import zipfile
from pathlib import Path

class HFVectorStorage:
    """Manages vector database storage on Hugging Face Hub."""
    
    def __init__(self, repo_id: str, token: str = None):
        """
        Initialize HF storage manager.
        """
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi()
        
    def upload_vector_db(self, local_db_path: str = "./chroma_db", commit_message: str = None):
        """
        Upload ChromaDB to Hugging Face Hub.
        
        Args:
            local_db_path: Path to local ChromaDB directory
            commit_message: Optional commit message
        """
        print(f"📤 Uploading vector database to {self.repo_id}...")
        
        # Create zip file
        zip_path = "chroma_db.zip"
        print(f"  📦 Zipping {local_db_path}...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(local_db_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, local_db_path)
                    zipf.write(file_path, arcname)
        
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  ✅ Zip created: {zip_size_mb:.1f} MB")
        
        # Upload to HF Hub
        try:
            self.api.upload_file(
                path_or_fileobj=zip_path,
                path_in_repo="chroma_db.zip",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message or "Update vector database"
            )
            print(f"  ✅ Uploaded to Hugging Face Hub!")
            print(f"  🔗 View at: https://huggingface.co/datasets/{self.repo_id}")
        finally:
            # Cleanup
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
    def download_vector_db(self, local_db_path: str = "./chroma_db", force: bool = False):
        """
        Download ChromaDB from Hugging Face Hub.
        
        Args:
            local_db_path: Where to extract the database
            force: Force re-download even if exists
        """
        if os.path.exists(local_db_path) and not force:
            print(f"✅ Vector database already exists at {local_db_path}")
            return
        
        print(f"📥 Downloading vector database from {self.repo_id}...")
        
        try:
            # Download zip file
            zip_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="chroma_db.zip",
                repo_type="dataset",
                token=self.token
            )
            
            print(f"  ✅ Downloaded from Hugging Face Hub")
            
            # Extract
            print(f"  📦 Extracting to {local_db_path}...")
            
            # Remove existing directory if force=True
            if os.path.exists(local_db_path):
                import shutil
                shutil.rmtree(local_db_path)
            
            # Extract to temp location first
            temp_extract = local_db_path + "_temp"
            if os.path.exists(temp_extract):
                import shutil
                shutil.rmtree(temp_extract)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_extract)
            
            # Check if there's a nested chroma_db folder
            nested_path = os.path.join(temp_extract, "chroma_db")
            if os.path.exists(nested_path):
                # Move the nested folder to the target location
                import shutil
                shutil.move(nested_path, local_db_path)
                shutil.rmtree(temp_extract)
            else:
                # No nesting, just rename temp to target
                import shutil
                shutil.move(temp_extract, local_db_path)
            
            print(f"  ✅ Vector database ready at {local_db_path}")
            
        except Exception as e:
            print(f"  ❌ Error downloading: {e}")
            raise
            
    def get_db_info(self):
        """Get information about the database on HF Hub."""
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            if "chroma_db.zip" in files:
                # Get file info
                file_info = self.api.get_paths_info(
                    repo_id=self.repo_id,
                    paths=["chroma_db.zip"],
                    repo_type="dataset"
                )
                return file_info[0]
            else:
                return None
        except Exception as e:
            print(f"Error getting DB info: {e}")
            return None
