"""
Benchmark class for AgentFlow - loads and evaluates benchmark datasets.
Updated with SQuAD-style evaluation metrics (Normalization, Counter-based F1).
"""

import json
import os
import string
import re
import difflib
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
# Note: Avoid importing OpenAI SDK at module import time to keep
#       this module usable without that dependency. We import lazily
#       inside _llm_judgement only when needed.
# tqdm is optional; provide a graceful fallback if not installed
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# SQuAD-style Evaluation Helper Functions
# ==========================================

def _parse_json_field(value: Any) -> Any:
    """Parse JSON-encoded dataset fields while leaving plain strings unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _ensure_str_list(value: Any) -> List[str]:
    """Normalize a string/list/JSON-string image field into a list of strings."""
    parsed = _parse_json_field(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed]
    return []

def normalize_answer(s):
    """
    对答案进行标准化处理：
    1. 移除冠词 (a, an, the)
    2. 修复多余空白
    3. 移除标点符号
    4. 转换为小写
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def bool_mapping(s):
    """
    将布尔字符串映射为 yes/no，用于统一格式
    """
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

def compute_exact_match(prediction, ground_truth):
    """
    严格精确匹配：标准化后的字符串必须完全相等
    """
    return (normalize_answer(bool_mapping(prediction)) == normalize_answer(bool_mapping(ground_truth)))

def compute_f1_score(prediction, ground_truth):
    """
    计算 F1 分数，包含 Precision 和 Recall。
    针对 'yes', 'no', 'noanswer' 有特殊惩罚机制。
    Returns: (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0.0, 0.0, 0.0)

    # 特殊处理：如果预测或真实值是 yes/no/noanswer 且不匹配，直接给 0 分
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 构建词袋并统计重叠词数
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    # 如果没有重叠词
    if num_same == 0:
        return ZERO_METRIC

    # 计算 Precision, Recall, F1
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

# ==========================================

@dataclass
class BenchmarkItem:
    """Represents a single benchmark item with question and answer."""
    id: str
    question: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a prediction against ground truth."""
    item_id: str
    question: str
    ground_truth: str
    prediction: str
    score: float
    metric_name: str
    details: Optional[Dict[str, Any]] = None


class Benchmark(ABC):
    """
    Abstract base class for benchmark datasets.
    
    This class provides functionality for:
    - Loading benchmark data from JSON/JSONL files
    - Evaluating predictions against ground truth
    - Computing various evaluation metrics
    - Managing benchmark datasets
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None):
        """
        Initialize the benchmark.
        
        Args:
            data_path: Path to the benchmark data file
            name: Name of the benchmark
            description: Description of the benchmark
        """
        self.name = name or "Unknown Benchmark"
        self.description = description or ""
        self.data_path = data_path
        self.items: List[BenchmarkItem] = []
        self.evaluation_results: List[EvaluationResult] = []
        # Track current file and dataset root for resolving relative assets
        self._current_file_path: Optional[str] = None
        self._dataset_root: Optional[str] = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """
        Load benchmark data from a file or directory.
        
        Args:
            data_path: Path to the data file (JSON/JSONL) or a directory containing them.
        """
        abs_data_path = os.path.abspath(data_path)
        self.data_path = abs_data_path
        # always start fresh before loading new data
        self.items = []
        
        if not os.path.exists(abs_data_path):
            raise FileNotFoundError(f"Data file not found: {abs_data_path}")

        # Remember dataset root for relative asset resolution
        self._dataset_root = abs_data_path if os.path.isdir(abs_data_path) else os.path.dirname(abs_data_path)
        
        # Directory mode: load all json/jsonl files inside
        if os.path.isdir(abs_data_path):
            files: List[str] = []
            for root, _, fnames in os.walk(abs_data_path):
                for fname in fnames:
                    if fname.endswith(".jsonl") or fname.endswith(".json"):
                        files.append(os.path.join(root, fname))
            files.sort()

            if not files:
                raise ValueError(f"No .json or .jsonl files found in directory: {abs_data_path}")

            for fpath in files:
                try:
                    self._load_with_context(fpath)
                except Exception as exc:
                    print(f"Warning: Failed to load {fpath}: {exc}")
                    continue

            print(f"Loaded {len(self.items)} items from directory {abs_data_path} ({len(files)} files)")
            return
        
        # File mode
        if abs_data_path.endswith('.jsonl'):
            self._load_with_context(abs_data_path)
        elif abs_data_path.endswith('.json'):
            self._load_with_context(abs_data_path)
        else:
            raise ValueError(f"Unsupported file format: {abs_data_path}")
        
        print(f"Loaded {len(self.items)} items from {abs_data_path}")

    def _load_with_context(self, file_path: str) -> None:
        """Load a JSON/JSONL file while tracking the current file path."""
        prev = self._current_file_path
        self._current_file_path = file_path
        try:
            if file_path.endswith(".jsonl"):
                self._load_jsonl(file_path)
            elif file_path.endswith(".json"):
                self._load_json(file_path)
        finally:
            self._current_file_path = prev
    
    def _load_jsonl(self, file_path: str):
        """Load data from JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    item = self._parse_item(data, line_num)
                    self.items.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    def _load_json(self, file_path: str):
        """Load data from JSON file (with lightweight repair for common formatting errors)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()
            data = self._parse_json_content(raw, file_path)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of items
            for i, item_data in enumerate(data):
                item = self._parse_item(item_data, i + 1)
                self.items.append(item)
        elif isinstance(data, dict):
            # Single item or structured data
            if 'items' in data:
                # Structured format with items array
                for i, item_data in enumerate(data['items']):
                    item = self._parse_item(item_data, i + 1)
                    self.items.append(item)
            else:
                # Single item
                item = self._parse_item(data, 1)
                self.items.append(item)

    def _parse_json_content(self, raw: str, file_path: str) -> Any:
        """Parse JSON content, optionally repairing simple formatting issues."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            repaired = self._repair_json_content(raw, file_path)
            if repaired is None or repaired == raw:
                raise
            try:
                return json.loads(repaired)
            except Exception:
                # If repair still fails, raise original for clearer context
                raise exc

    def _repair_json_content(self, raw: str, file_path: str) -> Optional[str]:
        """
        Attempt minimal, safe repairs to JSON content:
        - Fill missing values written as `: ,` with null
        - Insert missing commas between adjacent object fields

        Returns repaired string or None when no safe fix is applied.
        """
        fixed = raw
        applied = 0

        # Fix patterns like `"ReverseImageSearch": ,`
        pattern_missing_value = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*,')
        new_fixed = pattern_missing_value.sub(r'"\1": null,', fixed)
        if new_fixed != fixed:
            fixed = new_fixed
            applied += 1

        # Insert missing commas between object fields split by newline
        # e.g., `"description": "..."` (newline) `"key_info": "..."`
        pattern_missing_comma = re.compile(r'(".*?")(\s*\n\s*)"([A-Za-z0-9_]+)"\s*:')  # noqa: E501
        new_fixed = pattern_missing_comma.sub(r'\1,\2"\3":', fixed)
        if new_fixed != fixed:
            fixed = new_fixed
            applied += 1

        if applied:
            print(f"Warning: Repaired JSON formatting ({applied} fix(es)) in {file_path}")
            return fixed
        return None
    
    def _parse_item(self, data: Dict[str, Any], line_num: int) -> BenchmarkItem:
        """
        Parse a single item from the data.
        Override this method for custom parsing logic.

        This method now supports flexible metadata handling:
        - Keeps common fields (id, question, answer) as top-level attributes
        - Puts all other fields into metadata dictionary
        - Ensures answer is always a string (converts int/float/etc. to str)
        - Handles various metadata structures (nested dicts, lists, etc.)

        Args:
            data: Raw item data
            line_num: Line number for error reporting

        Returns:
            Parsed BenchmarkItem
        """
        # Define common/reserved field names that should NOT go into metadata
        # These are standard fields used across different benchmark types
        reserved_fields = {'id', 'question', 'answer'}

        # Extract required fields
        item_id = data.get('id', f'item_{line_num}')
        question = data.get('question', '')

        # Extract and normalize answer field (ensure it's always a string)
        answer_raw = data.get('answer', '')
        if isinstance(answer_raw, str):
            answer = answer_raw
        elif answer_raw is None:
            answer = ''
        else:
            # Convert int, float, bool, etc. to string
            answer = str(answer_raw)

        # Parse common JSON-string dataset fields before metadata extraction.
        for field_name in ("metadata", "sub_goals", "image", "image_paths"):
            if field_name in data:
                data[field_name] = _parse_json_field(data[field_name])

        # Extract metadata (all other fields that are not reserved)
        # This supports various metadata structures:
        # - BrowseComp-V3: domain, level, difficulty, sub-goals, image metadata
        # - Custom: any other fields
        #
        # Additionally, if an inner key named "metadata" exists and is a dict,
        # we flatten its contents into the top-level metadata for easier access.
        # Example:
        #   {
        #     "id": "...", "question": "...", "answer": "...",
        #     "images": [...],
        #     "metadata": {"domain": {"d1": "Science"}}
        #   }
        # Becomes:
        #   item.metadata == {"images": [...], "domain": {"d1": "Science"}}
        metadata: Dict[str, Any] = {}
        for k, v in data.items():
            if k in reserved_fields:
                continue
            # Flatten nested metadata dict if present
            if k == 'metadata' and isinstance(v, dict):
                # Keep existing keys if collision occurs; nested keys won't override
                for inner_k, inner_v in v.items():
                    if inner_k not in metadata:
                        metadata[inner_k] = inner_v
                continue
            metadata[k] = v

        if 'images' not in metadata:
            image_refs: List[str] = []
            image_refs.extend(_ensure_str_list(data.get('image_paths')))
            image_refs.extend(_ensure_str_list(data.get('image')))
            nested_metadata = data.get('metadata')
            if isinstance(nested_metadata, dict):
                image_refs.extend(_ensure_str_list(nested_metadata.get('images')))
            if image_refs:
                seen: List[str] = []
                for image_ref in image_refs:
                    if image_ref not in seen:
                        seen.append(image_ref)
                metadata['images'] = seen

        # Normalize image paths inside metadata to match current folder structure
        try:
            if 'images' in metadata:
                current_dir = os.path.dirname(self._current_file_path) if self._current_file_path else None
                metadata['images'] = self._resolve_image_paths(metadata['images'], file_dir=current_dir)
        except Exception:
            # Keep original images on any resolution error to avoid breaking loads
            pass

        return BenchmarkItem(
            id=item_id,
            question=question,
            answer=answer,
            metadata=metadata
        )

    def _resolve_image_paths(self, images: Any, file_dir: Optional[str] = None) -> Any:
        """
        Resolve image paths in metadata to align with the dataset's folder layout.

        Strategy (non-destructive):
        - Absolute paths: keep as-is
        - Relative paths: try multiple base directories, pick the first existing
            1) Directory of the current data file
            2) Dataset root (directory passed to Benchmark in folder mode)
            3) Parent directory of the data file (useful for paths like "./Science/..")
            4) CWD fallback for already-correct relative paths
          Also attempts common nested layouts like "<root>/<split>/image/<file>"
        - If no candidate exists on disk, keep the original string

        Supports both list[str] and single str.
        """
        # Normalize to list for unified processing
        is_list_input = True
        if isinstance(images, str):
            images_list = [images]
            is_list_input = False
        elif isinstance(images, (list, tuple)):
            images_list = list(images)
        else:
            # Unknown structure; return as-is
            return images

        data_dir = file_dir or (os.path.dirname(self.data_path) if self.data_path and not os.path.isdir(self.data_path) else '')
        data_parent = os.path.dirname(data_dir) if data_dir else ''
        dataset_root = self._dataset_root or ''

        def _append_unique(seq: List[str], value: str):
            if value and value not in seq:
                seq.append(value)

        def _relative_candidates(path_str: str) -> List[str]:
            candidates: List[str] = []
            normalized = os.path.normpath(path_str)
            trimmed = os.path.normpath(path_str.lstrip("./"))

            _append_unique(candidates, normalized)
            _append_unique(candidates, trimmed)

            parts = trimmed.split(os.sep)
            if parts:
                # Insert "image" after the first component (e.g., Science/image/xxx.jpg)
                if parts[0] != "image":
                    with_image = os.path.normpath(os.path.join(parts[0], "image", *parts[1:]))
                    _append_unique(candidates, with_image)
                    prepend_image = os.path.normpath(os.path.join("image", *parts))
                    _append_unique(candidates, prepend_image)
            return candidates

        resolved = []
        for p in images_list:
            if not isinstance(p, str):
                resolved.append(p)
                continue

            p_stripped = p.strip()
            # Absolute path
            if os.path.isabs(p_stripped):
                resolved.append(os.path.abspath(p_stripped) if os.path.exists(p_stripped) else p_stripped)
                continue

            bases: List[str] = []
            for b in [data_dir, dataset_root, data_parent]:
                if b and b not in bases:
                    bases.append(b)

            chosen = None
            for rel in _relative_candidates(p_stripped):
                # Try provided bases first
                for base in bases:
                    candidate = os.path.abspath(os.path.join(base, rel))
                    if os.path.exists(candidate):
                        chosen = candidate
                        break
                if chosen:
                    break
                # Finally, try current working directory for already-correct relative paths
                if os.path.exists(rel):
                    chosen = os.path.abspath(rel)
                    break

            resolved.append(chosen if chosen is not None else p)

        return resolved if is_list_input else resolved[0]

    def get_item(self, item_id: str) -> Optional[BenchmarkItem]:
        """Get a specific item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def get_items(self) -> List[BenchmarkItem]:
        """Get all items."""
        return self.items.copy()
    
    def get_questions(self) -> List[str]:
        """Get all questions."""
        return [item.question for item in self.items]
    
    def get_answers(self) -> List[str]:
        """Get all ground truth answers."""
        return [item.answer for item in self.items]

    def evaluate(self, 
                 predictions: Union[Dict[str, str], List[str]], 
                 metric: str = "exact_match",
                 concurrent: bool = True,
                 max_workers: Optional[int] = None,
                 **kwargs) -> List[EvaluationResult]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: Dict mapping item_id to prediction, or list of predictions in order
            metric: Evaluation metric to use
            concurrent: Evaluate in parallel when True
            max_workers: Optional override for thread pool size
            **kwargs: Additional arguments for the metric function
            
        Returns:
            List of evaluation results
        """
        if isinstance(predictions, list):
            if len(predictions) != len(self.items):
                raise ValueError(
                    f"Number of predictions ({len(predictions)}) "
                    f"doesn't match number of items ({len(self.items)})"
                )
            pred_dict = {item.id: pred for item, pred in zip(self.items, predictions)}
        else:
            pred_dict = predictions
        
        missing_ids = [item.id for item in self.items if item.id not in pred_dict]
        for missing_id in missing_ids:
            print(f"Warning: No prediction for item {missing_id}")

        items_to_evaluate = [
            (index, item) for index, item in enumerate(self.items) if item.id in pred_dict
        ]

        def evaluate_single(index_item_pair):
            index, item = index_item_pair
            prediction = pred_dict[item.id]
            score = self._compute_metric(
                item.answer,
                prediction,
                metric,
                question=item.question,
                **kwargs,
            )
            result = EvaluationResult(
                item_id=item.id,
                question=item.question,
                ground_truth=item.answer,
                prediction=prediction,
                score=score,
                metric_name=metric,
                details=self._get_metric_details(
                    item.answer, prediction, metric, **kwargs
                ),
            )
            return index, result

        if not concurrent or len(items_to_evaluate) <= 1:
            results = []
            for index_item_pair in tqdm(items_to_evaluate):
                _, result = evaluate_single(index_item_pair)
                results.append(result)
        else:
            results_dict: Dict[int, EvaluationResult] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(evaluate_single, index_item_pair): index_item_pair[1]
                    for index_item_pair in items_to_evaluate
                }
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
                    item = future_to_item[future]
                    try:
                        index, result = future.result()
                        results_dict[index] = result
                    except Exception as exc:
                        raise RuntimeError(f"Error evaluating item {item.id}") from exc
            results = [results_dict[index] for index, _ in items_to_evaluate]
        
        self.evaluation_results = results
        return results
    
    def _compute_metric(self, 
                       ground_truth: str, 
                       prediction: str, 
                       metric: str, 
                       question: Optional[str] = None,
                       **kwargs) -> float:
        """Compute a specific metric between ground truth and prediction."""
        metric_func = self._get_metric_function(metric)
        if metric == 'llm_judgement':
            return metric_func(question, ground_truth, prediction, **kwargs)
        else:
            return metric_func(ground_truth, prediction, **kwargs)
    
    def _get_metric_function(self, metric: str) -> Callable:
        """Get the metric function by name."""
        metric_functions = {
            'exact_match': self._exact_match,
            'f1_score': self._f1_score,
            'bleu_score': self._bleu_score,
            'rouge_score': self._rouge_score,
            'similarity': self._similarity,
            'contains_answer': self._contains_answer,
            'numeric_match': self._numeric_match,
            'llm_judgement': self._llm_judgement
        }
        
        if metric not in metric_functions:
            raise ValueError(f"Unknown metric: {metric}. "
                           f"Available metrics: {list(metric_functions.keys())}")
        
        return metric_functions[metric]
    
    def _exact_match(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """Exact string match using SQuAD normalization."""
        return 1.0 if compute_exact_match(prediction, ground_truth) else 0.0
    
    def _f1_score(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """F1 score based on word overlap using SQuAD logic."""
        f1, _, _ = compute_f1_score(prediction, ground_truth)
        return f1
    
    def _bleu_score(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """Simple BLEU-like score based on n-gram overlap."""
        # Simple implementation - in practice, you'd use a proper BLEU library
        gt_words = ground_truth.lower().split()
        pred_words = prediction.lower().split()
        
        if not gt_words or not pred_words:
            return 0.0
        
        # 1-gram precision
        matches = 0
        for word in pred_words:
            if word in gt_words:
                matches += 1
        
        return matches / len(pred_words)
    
    def _rouge_score(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """Simple ROUGE-like score based on longest common subsequence."""
        # Simple implementation - in practice, you'd use a proper ROUGE library
        gt_words = ground_truth.lower().split()
        pred_words = prediction.lower().split()
        
        if not gt_words or not pred_words:
            return 0.0
        
        # LCS-based ROUGE-L
        lcs_length = self._lcs_length(gt_words, pred_words)
        recall = lcs_length / len(gt_words)
        precision = lcs_length / len(pred_words)
        
        if recall + precision == 0:
            return 0.0
        
        return 2 * (recall * precision) / (recall + precision)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _similarity(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """String similarity using difflib."""
        return difflib.SequenceMatcher(None, ground_truth, prediction).ratio()
    
    def _contains_answer(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """Check if prediction contains the ground truth answer."""
        # Use normalization here too for consistency
        norm_gt = normalize_answer(ground_truth)
        norm_pred = normalize_answer(prediction)
        return 1.0 if norm_gt in norm_pred else 0.0
    
    def _numeric_match(self, ground_truth: str, prediction: str, **kwargs) -> float:
        """Extract and compare numeric values."""
        gt_numbers = self._extract_numbers(ground_truth)
        pred_numbers = self._extract_numbers(prediction)
        
        if not gt_numbers and not pred_numbers:
            return 1.0
        if not gt_numbers or not pred_numbers:
            return 0.0
        
        # Check if any ground truth number appears in prediction
        for gt_num in gt_numbers:
            if any(abs(gt_num - pred_num) < 1e-6 for pred_num in pred_numbers):
                return 1.0
        
        return 0.0

    def _llm_judgement(self, question: str, labeled_answer: str, pred_answer: str, **kwargs) -> float:
        """Use LLM to judge the prediction."""
        # Use LLM to judge the prediction
        # Lazy import to avoid hard dependency when not using this metric
        try:
            import openai  # type: ignore
        except Exception:
            # If OpenAI SDK is not available, return 0.0 rather than crashing
            # so other metrics and dataset loading still function.
            return 0.0
        PROMPT = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""
        
        client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE", "")
            )
        
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        result = response.choices[0].message.content
        return float((result.strip() if result is not None else "") == "Correct")
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        # Simple regex to find numbers (including decimals)
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches if match]
    
    def _get_metric_details(self, 
                           ground_truth: str, 
                           prediction: str, 
                           metric: str, 
                           **kwargs) -> Dict[str, Any]:
        """Get additional details for a metric."""
        details = {}
        
        if metric == "f1_score":
            f1, precision, recall = compute_f1_score(prediction, ground_truth)
            details = {
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
        elif metric == "numeric_match":
            details = {
                "ground_truth_numbers": self._extract_numbers(ground_truth),
                "prediction_numbers": self._extract_numbers(prediction)
            }
        
        return details
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the benchmark."""
        if not self.evaluation_results:
            return {
                "name": self.name,
                "description": self.description,
                "total_items": len(self.items),
                "evaluation_status": "No evaluation performed"
            }
        
        scores = [result.score for result in self.evaluation_results]
        metric_name = self.evaluation_results[0].metric_name if self.evaluation_results else "unknown"
        
        return {
            "name": self.name,
            "description": self.description,
            "total_items": len(self.items),
            "evaluated_items": len(self.evaluation_results),
            "metric": metric_name,
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "perfect_matches": sum(1 for score in scores if score == 1.0)
        }
    
    def save_results(self, file_path: str):
        """Save evaluation results to a file."""
        results_data = {
            "benchmark_info": {
                "name": self.name,
                "description": self.description,
                "data_path": self.data_path,
                "total_items": len(self.items)
            },
            "evaluation_results": [
                {
                    "item_id": result.item_id,
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "prediction": result.prediction,
                    "score": result.score,
                    "metric_name": result.metric_name,
                    "details": result.details
                }
                for result in self.evaluation_results
            ],
            "summary": self.get_summary()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to {file_path}")
    
    def load_results(self, file_path: str):
        """Load evaluation results from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Load evaluation results
        self.evaluation_results = []
        for result_data in results_data.get("evaluation_results", []):
            result = EvaluationResult(
                item_id=result_data["item_id"],
                question=result_data["question"],
                ground_truth=result_data["ground_truth"],
                prediction=result_data["prediction"],
                score=result_data["score"],
                metric_name=result_data["metric_name"],
                details=result_data.get("details")
            )
            self.evaluation_results.append(result)
        
        print(f"Evaluation results loaded from {file_path}")

# Helper: quick Science dataset check
def _load_science_dataset_summary(science_dir: str) -> Dict[str, Any]:
    import glob
    science_jsons = sorted(glob.glob(os.path.join(science_dir, "*.json")))
    summary = {
        "science_dir": science_dir,
        "found_files": len(science_jsons),
        "loaded_files": 0,
        "total_images": 0,
        "images_found": 0,
    }

    for jf in science_jsons:
        try:
            benchmark = create_benchmark(data_path=jf, name="Science Loader Check")
        except Exception:
            continue

        if not benchmark.items:
            continue
        item = benchmark.items[0]
        imgs = (item.metadata or {}).get("images", [])
        if isinstance(imgs, str):
            imgs = [imgs]
        exists = [os.path.exists(p) for p in imgs]

        summary["loaded_files"] += 1
        summary["total_images"] += len(imgs)
        summary["images_found"] += sum(1 for x in exists if x)

    return summary




# Convenience functions for common use cases
def create_benchmark(data_path: str, 
                    name: Optional[str] = None,
                    description: Optional[str] = None) -> Benchmark:
    """Create a benchmark from a data file."""
    return Benchmark(data_path=data_path, name=name, description=description)


# Example usage: verify Science dataset loading and image path resolution
if __name__ == "__main__":
    import glob
    import os

    # Determine Science dataset directory relative to this file
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    science_dir = os.path.join(src_dir, "data", "Science")

    print("Verifying benchmark loading for:")
    print(f" - {science_dir}")

    science_jsons = sorted(glob.glob(os.path.join(science_dir, "*.json")))
    print(f"Found {len(science_jsons)} JSON file(s)")

    loaded_files = 0
    total_images = 0
    images_found = 0

    for jf in science_jsons:
        try:
            benchmark = create_benchmark(data_path=jf, name="Science Loader Check")
        except Exception as e:
            print(f"Load error: {jf}: {e}")
            continue

        print(f"\nLoaded {len(benchmark.items)} item(s) from {jf}")
        if not benchmark.items:
            continue

        item = benchmark.items[0]
        imgs = (item.metadata or {}).get("images", [])
        if isinstance(imgs, str):
            imgs = [imgs]
        exists = [os.path.exists(p) for p in imgs]

        loaded_files += 1
        total_images += len(imgs)
        images_found += sum(1 for x in exists if x)

        print(f"  - ID: {item.id}")
        print(f"  - Question: {item.question[:80]}{'...' if len(item.question) > 80 else ''}")
        print(f"  - Images: {imgs}")
        print(f"  - Exists: {exists}")

    print("\nSummary:")
    print(f"  Science JSONs found: {len(science_jsons)}")
    print(f"  Successfully loaded: {loaded_files}")
    print(f"  Total images referenced: {total_images}")
    print(f"  Images resolved on disk: {images_found}")
