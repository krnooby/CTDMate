# validators/checklist.py
from __future__ import annotations
import re, json
from typing import Tuple, List, Dict, Any, Optional

try:
    import yaml
except Exception:
    yaml = None  # 런타임에 미설치면 문자열 기반 최소 동작

__all__ = ["validate_checklist_yaml", "evaluate_checklist"]

# ──────────────────────────────────────────────────────────────────────────────
# YAML 로더
def _load_yaml(yaml_or_path: str) -> Dict[str, Any]:
    text = yaml_or_path
    try:
        # 파일 경로일 수 있음
        with open(yaml_or_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        pass
    if yaml:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            return {}
    # YAML 미사용 환경: 아주 단순한 JSON 호환만 허용
    try:
        return json.loads(text)
    except Exception:
        return {}

# ──────────────────────────────────────────────────────────────────────────────
# 스키마 점검(느슨하게)
def validate_checklist_yaml(yaml_or_path: str) -> Tuple[bool, List[str]]:
    y = _load_yaml(yaml_or_path)
    errs: List[str] = []

    # 허용 루트 키
    allowed = {
        "version", "required_keys", "forbidden_terms", "coverage",
        "conditional_rules", "severity"
    }
    unknown = [k for k in y.keys() if k not in allowed]
    if unknown:
        errs.append(f"unknown keys: {unknown}")

    if "required_keys" in y and not isinstance(y["required_keys"], list):
        errs.append("required_keys must be a list[str]")

    if "forbidden_terms" in y and not isinstance(y["forbidden_terms"], list):
        errs.append("forbidden_terms must be a list[str]")

    cov = y.get("coverage", {})
    if cov and not isinstance(cov, dict):
        errs.append("coverage must be an object")
    else:
        w = cov.get("weights", {})
        if w and not isinstance(w, dict):
            errs.append("coverage.weights must be an object(str->number)")
        thr = cov.get("min_coverage", None)
        if thr is not None and not isinstance(thr, (int, float)):
            errs.append("coverage.min_coverage must be number")

    cr = y.get("conditional_rules", [])
    if cr and not isinstance(cr, list):
        errs.append("conditional_rules must be a list")
    else:
        for i, r in enumerate(cr):
            if not isinstance(r, dict):
                errs.append(f"conditional_rules[{i}] must be object")
                continue
            # 허용 키
            for rk in r.keys():
                if rk not in {"if_any", "if_all", "require", "require_patterns", "note"}:
                    errs.append(f"conditional_rules[{i}]: unknown key '{rk}'")
            if "if_any" in r and not isinstance(r["if_any"], list):
                errs.append(f"conditional_rules[{i}].if_any must be list[str]")
            if "if_all" in r and not isinstance(r["if_all"], list):
                errs.append(f"conditional_rules[{i}].if_all must be list[str]")
            if "require" in r and not isinstance(r["require"], list):
                errs.append(f"conditional_rules[{i}].require must be list[str]")
            if "require_patterns" in r and not isinstance(r["require_patterns"], list):
                errs.append(f"conditional_rules[{i}].require_patterns must be list[str]")

    return (len(errs) == 0, errs)

# ──────────────────────────────────────────────────────────────────────────────
def _find_snippet(text: str, term: str, window: int = 80) -> str:
    i = text.find(term)
    if i < 0:
        return ""
    a = max(0, i - window)
    b = min(len(text), i + len(term) + window)
    return text[a:b].replace("\n", " ")

def _present(text: str, key: str) -> bool:
    return key in text

def _present_pattern(text: str, pat: str) -> bool:
    try:
        return re.search(pat, text, flags=re.I | re.M) is not None
    except re.error:
        return False

def evaluate_checklist(md: str, checklist_yaml: str) -> Dict[str, Any]:
    """
    md: 검사할 Markdown 전체
    checklist_yaml: 체크리스트 YAML 문자열 또는 파일 경로
    return: {"summary": {...}, "items": [...]}  (에이전트가 기대하는 형식)
    """
    cfg = _load_yaml(checklist_yaml)

    required = list(cfg.get("required_keys", []))
    forbidden = list(cfg.get("forbidden_terms", []))
    coverage_cfg = cfg.get("coverage", {}) or {}
    weights: Dict[str, float] = {k: float(v) for k, v in (coverage_cfg.get("weights", {}) or {}).items()}
    min_cov: Optional[float] = coverage_cfg.get("min_coverage", None)
    cond_rules = cfg.get("conditional_rules", []) or []

    items: List[Dict[str, Any]] = []
    total_w = 0.0
    gained = 0.0

    # 필수 키 커버리지
    for k in required:
        w = float(weights.get(k, 1.0))
        total_w += max(w, 0.0)
        ok = _present(md, k)
        if ok:
            gained += max(w, 0.0)
        items.append({
            "key": k,
            "status": "pass" if ok else "fail",
            "weight": w,
            "evidence": [_find_snippet(md, k)] if ok else [],
            "remediation": "" if ok else f"문서에 '{k}' 관련 내용을 추가하세요."
        })

    coverage = (gained / total_w) if total_w > 0 else (1.0 if not required else 0.0)

    # 금칙어
    forbidden_hits: List[str] = []
    for w in forbidden:
        if w and w in md:
            forbidden_hits.append(w)
            items.append({
                "key": f"forbidden:{w}",
                "status": "fail",
                "weight": 1.0,
                "evidence": [_find_snippet(md, w)],
                "remediation": f"금칙어 '{w}'를 제거하거나 근거 문구로 대체하세요."
            })

    # 조건부 규칙
    for i, rule in enumerate(cond_rules):
        triggered = False
        if rule.get("if_any"):
            triggered = any(_present(md, k) for k in rule["if_any"])
        if not triggered and rule.get("if_all"):
            triggered = all(_present(md, k) for k in rule["if_all"])

        if triggered:
            # require (문자열 포함)
            for need in rule.get("require", []) or []:
                ok = _present(md, need)
                items.append({
                    "key": f"rule[{i}]-require:{need}",
                    "status": "pass" if ok else "fail",
                    "weight": 1.0,
                    "evidence": [_find_snippet(md, need)] if ok else [],
                    "remediation": "" if ok else f"조건부 규칙에 따라 '{need}'를 포함해야 합니다."
                })
            # require_patterns (정규식)
            for pat in rule.get("require_patterns", []) or []:
                ok = _present_pattern(md, pat)
                items.append({
                    "key": f"rule[{i}]-pattern:{pat}",
                    "status": "pass" if ok else "fail",
                    "weight": 1.0,
                    "evidence": [],  # 패턴은 스니펫 생략
                    "remediation": "" if ok else f"조건부 규칙에 따라 패턴 '{pat}'에 해당하는 내용이 필요합니다."
                })

    status = "complete"
    if forbidden_hits:
        status = "incomplete"
    if min_cov is not None and coverage < float(min_cov):
        status = "incomplete"

    return {
        "summary": {
            "status": status,
            "coverage": round(coverage, 3),
            "min_coverage": float(min_cov) if min_cov is not None else None,
            "forbidden_hits": forbidden_hits
        },
        "items": items
    }
