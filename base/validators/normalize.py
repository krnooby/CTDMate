# validators/normalize.py
from __future__ import annotations
import re, json
from typing import Tuple, List, Dict, Any

try:
    import yaml
except Exception:
    yaml = None

__all__ = ["validate_normalization_yaml", "normalize_text"]

def _load_yaml(yaml_or_path: str) -> Dict[str, Any]:
    text = yaml_or_path
    try:
        with open(yaml_or_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        pass
    if yaml:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            return {}
    try:
        return json.loads(text)
    except Exception:
        return {}

def validate_normalization_yaml(yaml_or_path: str) -> Tuple[bool, List[str]]:
    y = _load_yaml(yaml_or_path)
    errs: List[str] = []

    allowed = {"replacements", "terms", "units", "number_format", "whitespace"}
    unknown = [k for k in y.keys() if k not in allowed]
    if unknown:
        errs.append(f"unknown keys: {unknown}")

    reps = y.get("replacements", [])
    if reps and not isinstance(reps, list):
        errs.append("replacements must be a list of {pattern, replace, flags?}")
    else:
        for i, r in enumerate(reps or []):
            if not isinstance(r, dict) or "pattern" not in r or "replace" not in r:
                errs.append(f"replacements[{i}] invalid")

    terms = y.get("terms", [])
    if terms and not isinstance(terms, list):
        errs.append("terms must be a list of {from, to}")
    else:
        for i, t in enumerate(terms or []):
            if not isinstance(t, dict) or "from" not in t or "to" not in t:
                errs.append(f"terms[{i}] invalid")

    units = y.get("units", {})
    if units and not isinstance(units, dict):
        errs.append("units must be an object: canonical -> {aliases:[]}")
    else:
        for canon, spec in (units or {}).items():
            if not isinstance(spec, dict) or not isinstance(spec.get("aliases", []), list):
                errs.append(f"units[{canon}] must have aliases list")

    nf = y.get("number_format", {})
    if nf and not isinstance(nf, dict):
        errs.append("number_format must be an object")
    else:
        if "decimals" in nf and not isinstance(nf["decimals"], int):
            errs.append("number_format.decimals must be int")

    ws = y.get("whitespace", {})
    if ws and not isinstance(ws, dict):
        errs.append("whitespace must be an object")
    else:
        if "normalize" in ws and not isinstance(ws["normalize"], bool):
            errs.append("whitespace.normalize must be bool")

    return (len(errs) == 0, errs)

def _apply_replacements(text: str, reps: List[Dict[str, Any]]) -> str:
    out = text
    for r in reps or []:
        pat = r.get("pattern", "")
        rep = r.get("replace", "")
        flags = 0
        fl = (r.get("flags") or "").lower()
        if "i" in fl: flags |= re.IGNORECASE
        if "m" in fl: flags |= re.MULTILINE
        if "s" in fl: flags |= re.DOTALL
        try:
            out = re.sub(pat, rep, out, flags=flags)
        except re.error:
            continue
    return out

def _apply_terms(text: str, terms: List[Dict[str, str]]) -> str:
    out = text
    for t in terms or []:
        frm = str(t.get("from", ""))
        to  = str(t.get("to", ""))
        if not frm:
            continue
        out = re.sub(rf"\b{re.escape(frm)}\b", to, out, flags=re.IGNORECASE)
    return out

def _apply_units(text: str, units: Dict[str, Any]) -> str:
    out = text
    for canon, spec in (units or {}).items():
        aliases = spec.get("aliases", []) or []
        for a in aliases:
            if not a:
                continue
            out = re.sub(rf"\b{re.escape(str(a))}\b", str(canon), out, flags=re.IGNORECASE)
    return out

def _apply_numbers(text: str, nf: Dict[str, Any]) -> str:
    if not nf:
        return text
    dec = int(nf.get("decimals", 0))
    if dec < 0:
        return text

    def _fmt(m: re.Match) -> str:
        s = m.group(0)
        try:
            v = float(s)
            return f"{v:.{dec}f}" if dec > 0 else f"{int(round(v))}"
        except Exception:
            return s

    # 순수 숫자 토큰만 변환(단어 경계)
    return re.sub(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", _fmt, text)

def _apply_whitespace(text: str, ws: Dict[str, Any]) -> str:
    if not (ws or {}).get("normalize", True):
        return text
    out = re.sub(r"[ \t]+", " ", text)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

def normalize_text(text: str, yaml_or_path: str) -> str:
    cfg = _load_yaml(yaml_or_path)
    out = text
    out = _apply_replacements(out, cfg.get("replacements", []) or [])
    out = _apply_terms(out, cfg.get("terms", []) or [])
    out = _apply_units(out, cfg.get("units", {}) or {})
    out = _apply_numbers(out, cfg.get("number_format", {}) or {})
    out = _apply_whitespace(out, cfg.get("whitespace", {}) or {})
    return out
