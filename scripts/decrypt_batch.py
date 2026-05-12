#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from encryption_utils import derive_key, decrypt_text


def _parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _ensure_str_list(value):
    value = _parse_json_field(value)
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _collect_image_paths(obj):
    paths = []
    paths.extend(_ensure_str_list(obj.get("image_paths")))
    paths.extend(_ensure_str_list(obj.get("image")))
    metadata = _parse_json_field(obj.get("metadata"))
    if isinstance(metadata, dict):
        paths.extend(_ensure_str_list(metadata.get("images")))
    seen = []
    for p in paths:
        if p not in seen:
            seen.append(p)
    return seen


def _copy_referenced_images(paths, images_root, output_dir):
    mapping = {}
    image_out = output_dir / "images"
    for rel in paths:
        src = images_root / rel
        if not src.is_file() and rel.startswith("data/"):
            src = images_root / rel[5:]
        if not src.is_file():
            continue
        dst = image_out / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mapping[rel] = f"images/{dst.name}"
    return mapping


def _rewrite_image_refs(obj, mapping):
    def rewrite(value):
        values = _ensure_str_list(value)
        out = [mapping.get(v, v) for v in values]
        return json.dumps(out, ensure_ascii=False) if isinstance(value, str) else out
    if "image_paths" in obj:
        obj["image_paths"] = rewrite(obj["image_paths"])
    if "image" in obj:
        values = _ensure_str_list(obj["image"])
        if values:
            obj["image"] = mapping.get(values[0], values[0])
    metadata = _parse_json_field(obj.get("metadata"))
    if isinstance(metadata, dict) and "images" in metadata:
        metadata["images"] = [mapping.get(v, v) for v in _ensure_str_list(metadata.get("images"))]
        obj["metadata"] = metadata


def main():
    parser = argparse.ArgumentParser(description="Batch decrypt BrowseComp-V3 JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-encrypted", action="store_true")
    parser.add_argument("--copy-images", default=None, help="Dataset root, e.g. data/raw")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    key = derive_key(Path(args.key_file).read_text(encoding="utf-8").strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0

    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if args.limit is not None and count >= args.limit:
                break
            obj = json.loads(line)
            if "encrypted_data" in obj:
                enc = obj.get("encrypted_data", {})
                question = decrypt_text(enc.get("question", {}), key)
                answer = decrypt_text(enc.get("answer", {}), key)
            elif "encrypted_question" in obj:
                question = decrypt_text(json.loads(obj.get("encrypted_question", "{}")), key)
                answer = decrypt_text(json.loads(obj.get("encrypted_answer", "{}")), key)
            else:
                print(f"Warning: Unrecognized format in record {total}, skipping...", file=sys.stderr)
                continue

            out_obj = obj.copy()
            out_obj["question"] = question
            out_obj["answer"] = answer
            if args.copy_images:
                mapping = _copy_referenced_images(_collect_image_paths(out_obj), Path(args.copy_images), out_dir)
                _rewrite_image_refs(out_obj, mapping)
            if not args.keep_encrypted:
                for key_name in ["encrypted_data", "encrypted_question", "encrypted_answer", "trajectory"]:
                    out_obj.pop(key_name, None)
            (out_dir / f"{obj.get('id')}.json").write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            count += 1
    print(f"Decrypted {count}/{total} samples and wrote to {out_dir}")


if __name__ == "__main__":
    main()
