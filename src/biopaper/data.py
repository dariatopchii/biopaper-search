import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def clean_text(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value or ""))).strip()


def load_papers(path=None):
    path = Path(path or ROOT / "data/papers.jsonl")
    records, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("id") or not row.get("title") or not row.get("abstract"):
            raise ValueError("Each paper needs id, title, and abstract")
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        records.append(row)
    if not records:
        raise ValueError("Corpus contains no papers")
    return records


def corpus_hash(papers):
    content = json.dumps(papers, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(content).hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_jsonl(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []
