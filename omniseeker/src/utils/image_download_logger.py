"""
Image Download Logger - Tracks download attempts and failures for analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _default_log_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "image_downloads"


class ImageDownloadLogger:
    """Logs image download attempts and provides failure statistics."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir or os.getenv("IMAGE_DOWNLOAD_LOG_DIR") or _default_log_dir())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.failures: List[Dict] = []
        self.successes: int = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_success(self, url: str) -> None:
        """Record a successful download."""
        self.successes += 1

    def log_failure(self, url: str, error: str, attempts: int) -> None:
        """Record a failed download with details."""
        self.failures.append({
            "url": url,
            "error": error,
            "attempts": attempts,
            "timestamp": datetime.now().isoformat(),
            "error_category": self._categorize_error(error)
        })

    def get_summary(self) -> Dict:
        """Generate download statistics summary."""
        total = self.successes + len(self.failures)
        error_categories = {}
        for f in self.failures:
            cat = f.get("error_category", "other")
            error_categories[cat] = error_categories.get(cat, 0) + 1

        return {
            "total_attempts": total,
            "successful": self.successes,
            "failed": len(self.failures),
            "success_rate": f"{(self.successes / total * 100):.1f}%" if total > 0 else "N/A",
            "error_breakdown": error_categories
        }

    def save_failure_log(self, filepath: Optional[str] = None) -> str:
        """Save detailed failure log to JSON file."""
        if not self.failures:
            return ""

        if filepath is None:
            filepath = self.log_dir / f"failures_{self.session_id}.json"
        else:
            filepath = Path(filepath)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "summary": self.get_summary(),
                "failures": self.failures
            }, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def _categorize_error(self, error: str) -> str:
        """Categorize error for statistics."""
        error_lower = error.lower()
        if "timeout" in error_lower:
            return "timeout"
        elif "ssl" in error_lower or "certificate" in error_lower:
            return "ssl_error"
        elif "404" in error or "not found" in error_lower:
            return "not_found"
        elif "403" in error or "forbidden" in error_lower:
            return "forbidden"
        elif "connection" in error_lower:
            return "connection_error"
        elif "proxy" in error_lower:
            return "proxy_error"
        else:
            return "other"

    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()
        print(f"\n{'='*50}")
        print("Image Download Statistics")
        print(f"{'='*50}")
        print(f"Total attempts: {summary['total_attempts']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']}")
        if summary['error_breakdown']:
            print("\nError breakdown:")
            for category, count in summary['error_breakdown'].items():
                print(f"  - {category}: {count}")
        print(f"{'='*50}\n")

    def reset(self) -> None:
        """Reset statistics for a new session."""
        self.failures = []
        self.successes = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
