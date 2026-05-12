import asyncio
import base64
import csv
import hashlib
import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Callable

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - lightweight test/runtime fallback
    boto3 = None

    class ClientError(Exception):
        pass

from ..cache import (
    TOOL_IMAGE_UPLOAD_PUBLIC_URL,
    UPLOAD_OBJECT_KEY_VERSION,
    build_upload_query,
    get_cache_service,
)
from .config.settings import Config


class CloudStorageService:
    """
    Service for handling image uploads to Cloudflare R2.
    """

    def __init__(self):
        self.config = Config()
        self._s3 = self._init_s3_client()
        self._bucket = self._get_required_setting(
            "CLOUDFLARE_R2_BUCKET_NAME", self.config.CLOUDFLARE_R2_BUCKET_NAME
        )
        self._public_domain = self._get_required_setting(
            "CLOUDFLARE_R2_PUBLIC_DOMAIN", self.config.CLOUDFLARE_R2_PUBLIC_DOMAIN
        ).rstrip("/")
        self._prefix = self.config.CLOUDFLARE_R2_PREFIX
        self._hash_map_path = self.config.UPLOADS_DIR / "upload_hash_map.json"
        self._hash_map = self._load_hash_map()

    def _get_required_setting(self, name: str, value: str) -> str:
        if not value:
            raise ValueError(f"{name} is not set.")
        return value

    def _init_s3_client(self):
        if boto3 is None:
            raise ImportError("boto3 is required for CloudStorageService")
        account_id = self._get_required_setting(
            "CLOUDFLARE_R2_ACCOUNT_ID", self.config.CLOUDFLARE_R2_ACCOUNT_ID
        )
        access_key = self._get_required_setting(
            "CLOUDFLARE_R2_ACCESS_KEY_ID", self.config.CLOUDFLARE_R2_ACCESS_KEY_ID
        )
        secret_key = self._get_required_setting(
            "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
            self.config.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
        )
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        return boto3.client(
            service_name="s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

    def _get_content_type(self, file_path: Path) -> str:
        guessed = mimetypes.guess_type(str(file_path))[0]
        return guessed or "application/octet-stream"

    def _calculate_sha256(self, file_path: Path) -> str:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _load_hash_map(self) -> Dict[str, Dict[str, str]]:
        if not self._hash_map_path.exists():
            return {}
        try:
            with open(self._hash_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_hash_map(self) -> None:
        self.config.create_directories()
        with open(self._hash_map_path, "w", encoding="utf-8") as f:
            json.dump(self._hash_map, f, ensure_ascii=False, indent=2)

    def _upload_logic_sync(self, file_path: Path, parent_id: str) -> Dict[str, str]:
        """
        Synchronous implementation of the upload logic chain (Cloudflare R2):
        Upload -> Build Public Direct Link
        """
        file_name = file_path.name
        file_hash = self._calculate_sha256(file_path)
        cached = self._hash_map.get(file_hash)
        if cached and cached.get("url") and cached.get("fileID"):
            return {
                "name": file_name,
                "fileID": cached["fileID"],
                "url": cached["url"],
            }

        object_name = self._build_object_name(file_path, parent_id, file_hash)
        content_type = self._get_content_type(file_path)

        try:
            self._s3.upload_file(
                str(file_path),
                self._bucket,
                object_name,
                ExtraArgs={"ContentType": content_type},
            )
        except ClientError as exc:
            raise RuntimeError(f"Upload to Cloudflare R2 failed: {exc}") from exc

        result = self._get_direct_link_result(file_name, object_name)
        self._hash_map[file_hash] = {
            "fileID": result["fileID"],
            "url": result["url"],
        }
        self._save_hash_map()
        return result

    def _build_object_name(self, file_path: Path, parent_id: str, file_hash: str) -> str:
        suffix = file_path.suffix
        unique_name = f"{file_hash}{suffix}"
        segments = [seg for seg in (self._prefix, parent_id, unique_name) if seg]
        return "/".join(segments)

    def _get_direct_link_result(self, file_name: str, file_id: str) -> Dict[str, str]:
        """Build Cloudflare R2 public direct-link URL"""
        base_url = self._public_domain
        if not file_id:
            raise ValueError("fileID is required to build direct link.")
        url = f"{base_url}/{file_id}"
        
        if not url:
            raise ValueError(f"Failed to retrieve direct link for fileID: {file_id}")
            
        return {
            "name": file_name,
            "fileID": str(file_id),
            "url": url,
            "public_url": url,
            "object_key": str(file_id),
        }

    async def upload_single_image(self, file_path: Path) -> Dict[str, str]:
        """
        Upload a single image file to cloud storage (Async Wrapper)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.config.SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        file_hash = self._calculate_sha256(file_path)
        content_type = self._get_content_type(file_path)
        cache = get_cache_service()
        options = {
            "provider": "cloudflare_r2",
            "bucket": self._bucket,
            "public_domain": self._public_domain,
            "prefix": self._prefix,
            "suffix": file_path.suffix.lower(),
            "content_type": content_type,
            "object_key_version": UPLOAD_OBJECT_KEY_VERSION,
            "__trace": {
                "local_path": str(file_path),
            },
        }

        async def _fetch_fresh(_: str, opts: Dict[str, str]) -> Dict[str, str]:
            last_error = None

            for attempt in range(1, self.config.MAX_RETRIES + 1):
                try:
                    result = await asyncio.to_thread(
                        self._upload_logic_sync,
                        file_path,
                        "",
                    )
                    if "public_url" not in result and result.get("url"):
                        result["public_url"] = result["url"]
                    if "object_key" not in result and result.get("fileID"):
                        result["object_key"] = result["fileID"]
                    return result

                except Exception as e:
                    last_error = e
                    print(
                        f"[WARN] Upload failed ({file_path.name}), attempt {attempt}/{self.config.MAX_RETRIES}: {e}"
                    )

                    if attempt < self.config.MAX_RETRIES:
                        await asyncio.sleep(self.config.RETRY_SLEEP)

            raise RuntimeError(
                f"Upload permanently failed for {file_path} — last error: {last_error}"
            )

        return await cache.aget_or_fetch(
            tool_name=TOOL_IMAGE_UPLOAD_PUBLIC_URL,
            query=build_upload_query(file_hash),
            options=options,
            real_fetch_func=_fetch_fresh,
        )

    async def upload_data_url(
        self,
        data_url: str,
        *,
        filename_prefix: str = "upload",
    ) -> Dict[str, str]:
        """Persist a data URL to a temp file, upload it, then clean up if configured."""
        if not isinstance(data_url, str) or not data_url.startswith("data:image"):
            raise ValueError("data_url must start with data:image")
        if "," not in data_url:
            raise ValueError("Invalid Base64 format: missing comma separator.")

        header, encoded = data_url.split(",", 1)
        mime_prefix = header.split(";")[0]
        if "/" in mime_prefix:
            file_ext = mime_prefix.split("/")[1]
        else:
            file_ext = "png"

        filename = f"{filename_prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.{file_ext}"
        cache_path = self.config.SEARCH_CACHE_DIR / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "wb") as f:
            f.write(base64.b64decode(encoded))

        try:
            print(f"[INFO] Auto-uploading Base64 image: {filename}")
            return await self.upload_single_image(cache_path)
        finally:
            if not self.config.KEEP_LOCAL_CACHE:
                if cache_path.exists():
                    cache_path.unlink()
            else:
                print(f"[DEBUG] Kept Base64 cache: {filename}")
    
    async def upload_multiple_images(self, file_paths: List[Path], 
                                   progress_callback: Optional[Callable] = None) -> List[Dict[str, str]]:
        """
        Upload multiple image files to cloud storage
        """
        results = []
        total_files = len(file_paths)
        
        for idx, file_path in enumerate(file_paths, 1):
            if progress_callback:
                progress_callback(idx, total_files, file_path.name)
            
            try:
                result = await self.upload_single_image(file_path)
                results.append(result)
                print(f"[OK] {file_path.name} -> {result['url']}")
            except Exception as e:
                print(f"[ERROR] Failed to upload {file_path.name}: {e}")
                results.append({
                    "name": file_path.name,
                    "fileID": "",
                    "url": "",
                    "error": str(e)
                })
        
        return results
    
    def save_upload_mapping(self, results: List[Dict[str, str]], 
                           csv_path: Optional[Path] = None,
                           json_path: Optional[Path] = None) -> None:
        """
        Save upload results to CSV and JSON files
        """
        if not csv_path:
            csv_path = self.config.LOGS_DIR / "upload_mapping.csv"
        if not json_path:
            json_path = self.config.LOGS_DIR / "upload_mapping.json"
        
        # Ensure log directory exists
        self.config.create_directories()
        
        # Save to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["name", "fileID", "url"]
            if any("error" in result for result in results):
                fieldnames.append("error")
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        # Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] Upload mapping saved to {csv_path} and {json_path}")
    
    def is_supported_image(self, file_path: Path) -> bool:
        """Check if file is a supported image type"""
        return file_path.suffix.lower() in self.config.SUPPORTED_IMAGE_EXTENSIONS
    
    def find_images_in_directory(self, directory: Path) -> List[Path]:
        """Find all supported image files in a directory"""
        if not directory.exists() or not directory.is_dir():
            return []
        
        images = []
        for file_path in directory.iterdir():
            if file_path.is_file() and self.is_supported_image(file_path):
                images.append(file_path)
        
        return sorted(images)
