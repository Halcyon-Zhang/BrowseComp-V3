import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(verbose=True, override=True)

class Config:
    @staticmethod
    def _int_env(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    # API Keys
    # Prefer Serper.dev key, but fall back to legacy SERPAPI key to ease transition
    SERPER_API_KEY = os.getenv("SERPER_API_KEY") or os.getenv("SERPAPI_API_KEY", "")
    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

    # Serper.dev Configuration
    SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev").rstrip("/")
    SERPER_SEARCH_URL = f"{SERPER_BASE_URL}/search"
    SERPER_IMAGES_URL = f"{SERPER_BASE_URL}/images"
    SERPER_LENS_URL = f"{SERPER_BASE_URL}/lens"  # Google Lens reverse image search

    # Jina Reader Configuration
    JINA_BASE = "https://r.jina.ai/"

    # === Ops Configuration (运维配置) ===
    # 1. 存储策略：控制裁切工具是返回 URL 还是本地路径。默认 'cloud'。
    #    Agent 不再感知此参数。
    STORAGE_MODE = os.getenv("SEARCH_STORAGE_MODE", "cloud").lower()

    # 2. 调试开关：控制是否保留中间临时文件 (裁切图、Base64转存图)。默认 False。
    #    设为 True 时方便排查花屏或裁切偏移问题。
    KEEP_LOCAL_CACHE = os.getenv("SEARCH_KEEP_CACHE", "False").lower() in ("true", "1", "yes")

    # 3. 图片消息格式：控制工具返回的图片拼接到 message 时使用 URL 还是 Base64
    #    - 'url': 直接使用图片 URL（默认，体积小）
    #    - 'base64': 下载图片并转为 base64（适用于需要后续反向图搜的场景）
    IMAGE_MESSAGE_FORMAT = os.getenv("SEARCH_IMAGE_MESSAGE_FORMAT", "url").lower()

    # File and Path Configuration
    SEARCH_V2_ROOT = Path(__file__).resolve().parent.parent
    OMNISEEKER_ROOT = SEARCH_V2_ROOT.parents[2]
    PROJECT_ROOT = SEARCH_V2_ROOT
    UPLOADS_DIR = SEARCH_V2_ROOT / "uploads"

    # 专用子目录，防止文件混杂
    CROPS_DIR = UPLOADS_DIR / "crops"          # 存放裁切产物
    SEARCH_CACHE_DIR = UPLOADS_DIR / "cache"   # 存放 Base64 解码后的临时文件

    LOGS_DIR = Path(os.getenv("SEARCH_LOG_DIR", OMNISEEKER_ROOT / "logs"))

    # Cloudflare R2 Configuration
    CLOUDFLARE_R2_ACCOUNT_ID = os.getenv("CLOUDFLARE_R2_ACCOUNT_ID", "")
    CLOUDFLARE_R2_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "")
    CLOUDFLARE_R2_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "")
    CLOUDFLARE_R2_BUCKET_NAME = os.getenv("CLOUDFLARE_R2_BUCKET_NAME", "")
    CLOUDFLARE_R2_PUBLIC_DOMAIN = os.getenv("CLOUDFLARE_R2_PUBLIC_DOMAIN", "").rstrip("/")
    CLOUDFLARE_R2_PREFIX = os.getenv("CLOUDFLARE_R2_PREFIX", "").strip("/")

    # Image Configuration
    SUPPORTED_IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
    }

    # Search Configuration
    DEFAULT_SEARCH_RESULTS = _int_env.__func__("SEARCH_TOOLS_DEFAULT_K", 5)
    DEFAULT_IMAGE_RESULTS = _int_env.__func__("SEARCH_TOOLS_DEFAULT_IMAGE_K", DEFAULT_SEARCH_RESULTS)
    DEFAULT_REGION = os.getenv("SEARCH_TOOLS_DEFAULT_REGION", "us")

    # LLM Configuration
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.2
    MAX_SUMMARY_CHARS = 200

    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_SLEEP = 1.5

    # Timeout Configuration
    REQUEST_TIMEOUT = 60

    # ============ Image Download Configuration ============
    # Whether to prefetch images when search results return
    # Default: False (lazy loading), True = prefetch for reliability
    PREFETCH_IMAGES = os.getenv("SEARCH_PREFETCH_IMAGES", "False").lower() in ("true", "1", "yes")

    # Prefetch concurrency (only effective when PREFETCH_IMAGES=True)
    PREFETCH_CONCURRENCY = _int_env.__func__("SEARCH_PREFETCH_CONCURRENCY", 5)

    # Maximum retry attempts for image download
    IMAGE_DOWNLOAD_MAX_RETRIES = _int_env.__func__("IMAGE_DOWNLOAD_MAX_RETRIES", 3)

    # Retry delays in seconds (exponential backoff)
    IMAGE_DOWNLOAD_RETRY_DELAYS = [1, 2, 4]

    # Connection timeout and read timeout for image downloads (seconds)
    IMAGE_DOWNLOAD_CONNECT_TIMEOUT = _int_env.__func__("IMAGE_DOWNLOAD_CONNECT_TIMEOUT", 10)
    IMAGE_DOWNLOAD_READ_TIMEOUT = _int_env.__func__("IMAGE_DOWNLOAD_READ_TIMEOUT", 30)

    # Google thumbnail server rotation: try different tbn servers (0-3) when one fails
    # Number of full rounds to try all servers before giving up
    IMAGE_DOWNLOAD_SERVER_ROTATION_ROUNDS = _int_env.__func__("IMAGE_DOWNLOAD_SERVER_ROTATION_ROUNDS", 3)
    # List of Google thumbnail server suffixes to try
    GOOGLE_TBN_SERVERS = ["0", "1", "2", "3"]

    @classmethod
    def validate_required_keys(cls):
        """Validate that all required API keys are present"""
        required_keys = {
            "SERPER_API_KEY": cls.SERPER_API_KEY,
            "JINA_API_KEY": cls.JINA_API_KEY,
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
        }

        missing = [name for name, value in required_keys.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return True

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CROPS_DIR.mkdir(parents=True, exist_ok=True)
        cls.SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
