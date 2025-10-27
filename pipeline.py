# Detecting Citation Hallucinations — Minimal Hybrid Pipeline
# Stages: Exact Lookup → Fuzzy Retrieval → (Optional) LLM Verification
# Author: Vikranth Udandarao and Nipun Misra
# Usage:
#   python pipeline.py examples.jsonl
# Input format (JSONL): {"author": "First A.; Second B.", "title": "Paper title", "year": 2021, "venue": "Conf/Journal"}

from __future__ import annotations
import os, json, time, re, math, argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import requests
from dateutil.parser import parse as dateparse
from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ----------------------------
# Utilities
# ----------------------------

def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[\u2010-\u2015]", "-", s)  # normalize dashes
    return s.lower()

def _split_authors(s: str) -> List[str]:
    if not s: return []
    # Accept "First Last; Second Last" or "First Last, Second Last"
    s = s.replace(" and ", ";")
    parts = re.split(r"[;,]\s*", s)
    return [re.sub(r"\.", "", _norm(p)) for p in parts if p]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

# ----------------------------
# Data model
# ----------------------------

@dataclass
class Citation:
    author: str
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None

@dataclass
class Verdict:
    label: str                  # "valid" | "partially_valid" | "hallucinated"
    confidence: float           # 0..1
    matched_source: Optional[str] = None
    matched_id: Optional[str] = None
    debug: Dict[str, Any] = None

# ----------------------------
# Bibliography clients
# ----------------------------

class CrossrefClient:
    BASE = "https://api.crossref.org/works"

    def search(self, title: str, first_author: Optional[str]) -> List[Dict[str, Any]]:
        q = title
        params = {"query.bibliographic": q, "rows": 20}
        if first_author:
            params["query.author"] = first_author
        r = requests.get(self.BASE, params=params, timeout=20)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        return items

    @staticmethod
    def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
        title = it.get("title", [""])[0] if it.get("title") else ""
        year = None
        if "issued" in it and "date-parts" in it["issued"] and it["issued"]["date-parts"]:
            year = it["issued"]["date-parts"][0][0]
        authors = []
        for a in it.get("author", []) or []:
            nm = " ".join([a.get("given",""), a.get("family","")]).strip()
            if nm: authors.append(nm)
        venue = it.get("container-title", [""])[0] if it.get("container-title") else ""
        return {
            "id": it.get("DOI"),
            "source": "crossref",
            "title": title,
            "year": year,
            "authors": authors,
            "venue": venue
        }

class OpenAlexClient:
    BASE = "https://api.openalex.org/works"

    def search(self, title: str) -> List[Dict[str, Any]]:
        params = {"search": title, "per_page": 25}
        r = requests.get(self.BASE, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("results", [])

    @staticmethod
    def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
        title = it.get("title", "")
        year = it.get("publication_year")
        authors = [a.get("author", {}).get("display_name","") for a in it.get("authorships", [])]
        host = it.get("host_venue", {}) or {}
        venue = host.get("display_name","")
        return {
            "id": it.get("id"),
            "source": "openalex",
            "title": title,
            "year": year,
            "authors": authors,
            "venue": venue
        }

class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, title: str) -> List[Dict[str, Any]]:
        params = {
            "query": title,
            "limit": 20,
            "fields": "title,year,authors,venue,externalIds"
        }
        r = requests.get(self.BASE, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("data", [])

    @staticmethod
    def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
        title = it.get("title","")
        year = it.get("year")
        authors = [a.get("name","") for a in it.get("authors", [])]
        venue = it.get("venue","")
        return {
            "id": (it.get("externalIds") or {}).get("DOI") or it.get("paperId"),
            "source": "semanticscholar",
            "title": title,
            "year": year,
            "authors": authors,
            "venue": venue
        }

# ----------------------------
# Core detector
# ----------------------------

class Detector:
    def __init__(self, enable_llm: bool = False, openai_model: str = "gpt-4o-mini"):
        self.crossref = CrossrefClient()
        self.openalex = OpenAlexClient()
        self.semsch = SemanticScholarClient()
        self.enable_llm = enable_llm
        self.openai_model = openai_model

        if enable_llm:
            try:
                from openai import OpenAI
                self._openai = OpenAI()
            except Exception as e:
                raise RuntimeError("Install openai and set OPENAI_API_KEY to use LLM verification.") from e

    # ---- Stage 1: Exact lookup (high precision) ----
    def exact_lookup(self, c: Citation) -> Optional[Dict[str, Any]]:
        title_norm = _norm(c.title)
        first_author = _split_authors(c.author)[:1]
        first_author = first_author[0] if first_author else None

        candidates = []
        # Crossref
        try:
            cr = self.crossref.search(c.title, first_author)
            candidates += [CrossrefClient.normalize_item(x) for x in cr]
        except Exception:
            pass
        # OpenAlex
        try:
            oa = self.openalex.search(c.title)
            candidates += [OpenAlexClient.normalize_item(x) for x in oa]
        except Exception:
            pass
        # Semantic Scholar
        try:
            ss = self.semsch.search(c.title)
            candidates += [SemanticScholarClient.normalize_item(x) for x in ss]
        except Exception:
            pass

        best = None
        best_score = 0.0
        for it in candidates:
            title_score = fuzz.token_set_ratio(title_norm, _norm(it["title"])) / 100.0
            auth_score = jaccard(_split_authors(c.author), [_norm(a) for a in it["authors"]])
            year_match = 1.0 if (c.year and it["year"] and int(c.year) == int(it["year"])) else 0.0
            score = 0.6*title_score + 0.3*auth_score + 0.1*year_match
            if score > best_score:
                best, best_score = it, score

        # Thresholds tuned to prefer precision
        if best and best_score >= 0.92:
            best["_score"] = best_score
            return best
        return None

    # ---- Stage 2: Fuzzy retrieval (raise recall) ----
    def fuzzy_candidates(self, c: Citation) -> List[Dict[str, Any]]:
        # Gather a wider pool
        pool = []
        try:
            cr = self.crossref.search(c.title, None)
            pool += [CrossrefClient.normalize_item(x) for x in cr]
        except Exception:
            pass
        try:
            oa = self.openalex.search(c.title)
            pool += [OpenAlexClient.normalize_item(x) for x in oa]
        except Exception:
            pass
        try:
            ss = self.semsch.search(c.title)
            pool += [SemanticScholarClient.normalize_item(x) for x in ss]
        except Exception:
            pass

        # BM25 on titles+authors+venue
        docs = []
        for it in pool:
            text = " ".join([
                _norm(it["title"]),
                " ".join([_norm(a) for a in it["authors"]]),
                _norm(it["venue"] or "")
            ])
            docs.append(text)

        if not docs:
            return []

        bm25 = BM25Okapi([d.split() for d in docs])
        query = " ".join([_norm(c.title), _norm(c.author or ""), _norm(c.venue or "")]).split()
        scores = bm25.get_scores(query)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

        ranked = []
        for i in idx:
            it = pool[i]
            title_score = fuzz.token_set_ratio(_norm(c.title), _norm(it["title"])) / 100.0
            auth_score = jaccard(_split_authors(c.author), [_norm(a) for a in it["authors"]])
            year_close = 1.0 if (c.year and it["year"] and abs(int(c.year) - int(it["year"])) <= 1) else 0.0
            agg = 0.5*title_score + 0.3*auth_score + 0.2*year_close
            it = {**it, "_fuzzy": agg}
            ranked.append(it)

        ranked.sort(key=lambda x: x["_fuzzy"], reverse=True)
        return [r for r in ranked if r["_fuzzy"] >= 0.70][:5]

    # ---- Stage 3: Optional LLM verification (disambiguate) ----
    def llm_verify(self, c: Citation, candidates: List[Dict[str, Any]]) -> Verdict:
        if not self.enable_llm or not candidates:
            # Simple non-LLM fallback scoring
            best = max(candidates, key=lambda x: x["_fuzzy"])
            label, conf = ("valid", min(0.85, best["_fuzzy"])) if best["_fuzzy"] >= 0.85 else ("partially_valid", best["_fuzzy"])
            return Verdict(label=label, confidence=conf, matched_source=best["source"], matched_id=best["id"], debug={"picked": best})

        prompt = f"""
You are validating a bibliographic citation. Decide whether the query citation is:
- "valid" (title+authors match; year/venue consistent),
- "partially_valid" (same paper but some metadata off),
- "hallucinated" (no candidate is actually the same paper).

Return JSON with fields: label, confidence (0..1), chosen_index (or -1).

Query:
title="{c.title}"
authors="{c.author}"
year="{c.year}"
venue="{c.venue}"

Candidates:
{json.dumps([{k: v for k,v in it.items() if k in ["title","authors","year","venue","source","id"]} for it in candidates], indent=2)}
        """.strip()

        resp = self._openai.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        txt = resp.choices[0].message.content
        try:
            parsed = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception:
            # fallback heuristic if parsing fails
            best = max(candidates, key=lambda x: x["_fuzzy"])
            return Verdict(label="partially_valid", confidence=min(0.8, best["_fuzzy"]), matched_source=best["source"], matched_id=best["id"], debug={"llm_raw": txt})

        idx = parsed.get("chosen_index", -1)
        chosen = candidates[idx] if 0 <= idx < len(candidates) else None
        return Verdict(
            label=parsed.get("label", "partially_valid"),
            confidence=float(parsed.get("confidence", 0.6)),
            matched_source=chosen.get("source") if chosen else None,
            matched_id=chosen.get("id") if chosen else None,
            debug={"llm_raw": txt, "chosen": chosen}
        )

    # ---- Full pipeline ----
    def classify(self, c: Citation) -> Verdict:
        # Stage 1
        exact = self.exact_lookup(c)
        if exact:
            return Verdict(label="valid", confidence=0.95, matched_source=exact["source"], matched_id=exact["id"], debug={"exact": exact})

        # Stage 2
        cands = self.fuzzy_candidates(c)
        if not cands:
            return Verdict(label="hallucinated", confidence=0.9, matched_source=None, matched_id=None, debug={"reason": "no_candidates"})

        # Stage 3 (optional)
        return self.llm_verify(c, cands)

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="JSONL file of citations with fields: author, title, [year], [venue]")
    ap.add_argument("--llm", action="store_true", help="Enable LLM verification (set OPENAI_API_KEY).")
    args = ap.parse_args()

    det = Detector(enable_llm=args.llm)
    out_path = os.path.splitext(args.jsonl)[0] + ".verdicts.jsonl"

    with open(args.jsonl, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        for line in tqdm(f, desc="Classifying"):
            obj = json.loads(line)
            cit = Citation(**obj)
            verdict = det.classify(cit)
            rec = {**obj, **asdict(verdict)}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
