"""BioPaper Search: retrieval, human judgments, and honest evaluation."""
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(_root / '.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(_root / '.cache/torch'))
