"""Fetch an attributed local research corpus from Europe PMC. No credentials needed."""
import argparse
import datetime
import json
import time
import urllib.parse
import urllib.request
from biopaper.data import ROOT, clean_text, corpus_hash, write_jsonl

DEFAULT_QUERY = '("machine learning" OR "deep learning") AND (protein OR genomics OR "drug discovery" OR "single cell" OR CRISPR) AND OPEN_ACCESS:Y AND FIRST_PDATE:[2019-01-01 TO 2026-10-06]'


def fetch(query, limit):
    papers, seen, cursor = [], set(), "*"
    while len(papers) < limit:
        params = urllib.parse.urlencode({"query": query, "format": "json", "resultType": "core", "pageSize": min(100, limit), "cursorMark": cursor})
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + params
        response = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "BioPaperSearch/0.1 (educational retrieval experiment)"})
                with urllib.request.urlopen(req, timeout=45) as stream:
                    response = json.load(stream)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        batch = response.get("resultList", {}).get("result", [])
        if not batch:
            break
        for item in batch:
            abstract = clean_text(item.get("abstractText"))
            uid = item["source"] + ":" + item["id"]
            if len(abstract) < 100 or uid in seen:
                continue
            seen.add(uid)
            papers.append({"id": uid, "title": clean_text(item.get("title")), "abstract": abstract, "authors": item.get("authorString", ""), "year": item.get("pubYear", ""), "doi": item.get("doi", ""), "source": "Europe PMC", "url": 'https://europepmc.org/article/' + item["source"] + '/' + item["id"], "is_open_access": item.get("isOpenAccess") == "Y"})
            if len(papers) == limit:
                break
        next_cursor = response.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.4)
    if not papers:
        raise ValueError("The query returned no usable abstracts")
    return papers


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=300)
    p.add_argument('--query', default=DEFAULT_QUERY)
    args = p.parse_args()
    if not 1 <= args.limit <= 10000:
        p.error('limit must be between 1 and 10000')
    papers = fetch(args.query, args.limit)
    write_jsonl(ROOT / 'data/papers.jsonl', papers)
    (ROOT / 'data/corpus_manifest.json').write_text(json.dumps({"query": args.query, "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "count": len(papers), "sha256": corpus_hash(papers), "source": "https://europepmc.org/RestfulWebService", "note": "Local research corpus. Check individual article licences before redistribution."}, indent=2), encoding='utf-8')
    print(f'Saved {len(papers)} attributed abstracts.')
