# validators/rag_filters.py
from __future__ import annotations
import json
from typing import Tuple, List, Dict, Any, Optional

try:
    import yaml
except Exception:
    yaml = None

__all__ = ["validate_rag_filters_yaml", "evaluate_rag_filters"]

def _load_yaml(yaml_or_path: str) -> Dict[str, Any]:
    text = yaml_or_path
    try:
        with open(yaml_or_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        pass
    if yaml:
        try:
            return yaml.safe_load(text) | {}
        except Exception:
            return {}
    try:
        return json.loads(text)
    except Exception:
        return {}

def validate_rag_filters_yaml(yaml_or_path: str) -> Tuple[bool, List[str]]:
    y = _load_yaml(yaml_or_path)
    errs: List[str] = []

    allowed = {"sources", "sections", "ranking", "snippet", "dedup"}
    unknown = [k for k in y.keys() if k not in allowed]
    if unknown:
        errs.append(f"unknown keys: {unknown}")

    for key in ("sources", "sections"):
        v = y.get(key, {})
        if v and not isinstance(v, dict):
            errs.append(f"{key} must be an object")
        else:
            for kk in ("allow", "deny"):
                lst = v.get(kk)
                if lst is not None and not isinstance(lst, list):
                    errs.append(f"{key}.{kk} must be a list[str]")

    ranking = y.get("ranking", {})
    if ranking and not isinstance(ranking, dict):
        errs.append("ranking must be an object")
    else:
        mmr = ranking.get("mmr", {})
        if mmr and not isinstance(mmr, dict):
            errs.append("ranking.mmr must be an object")
        else:
            lam = mmr.get("lambda")
            if lam is not None and not isinstance(lam, (int, float)):
                errs.append("ranking.mmr.lambda must be number")
            for k in ("fetch_k", "k"):
                if k in mmr and not isinstance(mmr[k], int):
                    errs.append(f"ranking.mmr.{k} must be int")

    snip = y.get("snippet", {})
    if snip and not isinstance(snip, dict):
        errs.append("snippet must be an object")
    else:
        if "max_chars" in snip and not isinstance(snip["max_chars"], int):
            errs.append("snippet.max_chars must be int")
        if "min_similarity" in snip and not isinstance(snip["min_similarity"], (int, float)):
            errs.append("snippet.min_similarity must be number")

    dd = y.get("dedup", None)
    if dd is not None and not isinstance(dd, bool):
        errs.append("dedup must be boolean")

    return (len(errs) == 0, errs)

def evaluate_rag_filters(verified: Dict[str, Any], yaml_or_path: str) -> Tuple[bool, List[str]]:
    """
    verified: verify_rag(tool) 결과(JSON 딕셔너리)
      - 구조 예: {"ref_hits": {"섹션":[{"source","text","similarity"}, ...], ...}}
    return: (pass, issues[])
    """
    cfg = _load_yaml(yaml_or_path)
    issues: List[str] = []
    ref = (verified or {}).get("ref_hits", {}) or {}

    allow_sources = set((cfg.get("sources", {}) or {}).get("allow", []) or [])
    deny_sources  = set((cfg.get("sources", {}) or {}).get("deny", []) or [])
    allow_sections= set((cfg.get("sections", {}) or {}).get("allow", []) or [])
    deny_sections = set((cfg.get("sections", {}) or {}).get("deny", []) or [])

    snip = cfg.get("snippet", {}) or {}
    max_chars = int(snip.get("max_chars", 1000))
    min_sim  = float(snip.get("min_similarity", 0.0))
    dedup    = bool(cfg.get("dedup", False))

    seen = set()

    for sec, rows in ref.items():
        # 섹션 allow/deny
        if allow_sections and sec not in allow_sections:
            issues.append(f"section '{sec}' not allowed")
        if deny_sections and sec in deny_sections:
            issues.append(f"section '{sec}' is denied")

        for i, r in enumerate(rows or []):
            src = (r or {}).get("source") or ""
            txt = (r or {}).get("text") or ""
            sim = float((r or {}).get("similarity") or 0.0)

            # 소스 allow/deny
            if deny_sources and src in deny_sources:
                issues.append(f"ref[{sec}][{i}] denied source: {src}")
            if allow_sources and src not in allow_sources:
                issues.append(f"ref[{sec}][{i}] not in allowed sources: {src}")

            # 스니펫 길이
            if len(txt) > max_chars:
                issues.append(f"ref[{sec}][{i}] snippet too long: {len(txt)} > {max_chars}")

            # 유사도 하한
            if sim < min_sim:
                issues.append(f"ref[{sec}][{i}] similarity {sim:.3f} < {min_sim:.3f}")

            # 중복 제거 정책
            if dedup:
                key = (src, txt[:200])
                if key in seen:
                    issues.append(f"ref[{sec}][{i}] duplicated snippet from {src}")
                seen.add(key)

    return (len(issues) == 0, issues)
