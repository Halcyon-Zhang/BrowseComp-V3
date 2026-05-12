import importlib.util
from pathlib import Path


def test_release_layout_files_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "README.md",
        ".env.template",
        "requirements.txt",
        "docs/data.md",
        "docs/setup.md",
        "examples/baseline_rollout.py",
        "examples/eval_rollout_results.py",
        "omniseeker/run_bench.sh",
        "omniseeker/src/run_rollout.py",
        "scripts/download_dataset.py",
        "scripts/decrypt_dataset.py",
    ]:
        assert (root / rel).exists(), rel


def test_image_download_logs_default_under_omniseeker(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "omniseeker" / "src" / "utils" / "image_download_logger.py"
    spec = importlib.util.spec_from_file_location("image_download_logger", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("IMAGE_DOWNLOAD_LOG_DIR", raising=False)
    assert module._default_log_dir() == root / "omniseeker" / "logs" / "image_downloads"

    custom_log_dir = tmp_path / "image_downloads"
    monkeypatch.setenv("IMAGE_DOWNLOAD_LOG_DIR", str(custom_log_dir))
    logger = module.ImageDownloadLogger()
    assert logger.log_dir == custom_log_dir


def test_docs_match_release_paths():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    data_doc = (root / "docs" / "data.md").read_text(encoding="utf-8")
    setup_doc = (root / "docs" / "setup.md").read_text(encoding="utf-8")

    assert "printf '%s\\n'" in readme
    assert "results/baseline/gpt-4o/eval_gpt-4o.jsonl" in readme
    assert "data/samples/images/" in data_doc
    assert "data/images/..." not in data_doc
    assert "--key 'A_Visual" not in data_doc
    assert "--output data/decrypted_bcv3.json" not in data_doc
    assert "omniseeker/logs/image_downloads/" in setup_doc
    assert ".cache/search_cache.db" in setup_doc
