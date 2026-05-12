from pathlib import Path


def test_env_template_contains_runtime_knobs():
    text = Path(".env.template").read_text(encoding="utf-8")
    assert "SEARCH_IMAGE_MESSAGE_FORMAT" in text
    assert "MMDEEPSEARCH_CACHE_DB=.cache/search_cache.db" in text
    assert "SEARCH_LOG_DIR=omniseeker/logs" in text
