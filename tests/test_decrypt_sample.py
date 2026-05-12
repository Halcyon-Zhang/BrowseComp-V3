import json
import subprocess
import sys
from pathlib import Path

from scripts.encryption_utils import derive_key, encrypt_text


def test_decrypt_batch_rewrites_images(tmp_path):
    key_text = "A_Visual_Vertical_Verifiable_Benchmark_for_Multimodal_Browsing_Agents"
    key = derive_key(key_text)
    raw = tmp_path / "raw"
    (raw / "data" / "images").mkdir(parents=True)
    (raw / "data" / "images" / "sample.jpg").write_bytes(b"img")
    record = {
        "id": "sample",
        "image_paths": json.dumps(["data/images/sample.jpg"]),
        "encrypted_question": json.dumps(encrypt_text("question", key, b"0" * 16)),
        "encrypted_answer": json.dumps(encrypt_text("answer", key, b"1" * 16)),
    }
    input_path = raw / "data" / "train.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    key_file = tmp_path / "key.txt"
    key_file.write_text(key_text, encoding="utf-8")
    out = tmp_path / "samples"
    subprocess.check_call([sys.executable, "scripts/decrypt_batch.py", "--input", str(input_path), "--key-file", str(key_file), "--output-dir", str(out), "--copy-images", str(raw)])
    sample = json.loads((out / "sample.json").read_text(encoding="utf-8"))
    assert json.loads(sample["image_paths"]) == ["images/sample.jpg"]
    assert (out / "images" / "sample.jpg").exists()
