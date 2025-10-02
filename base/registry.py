from typing import Any, Dict, Optional, Callable
import os

# tool1
try:
    from tools.smartdoc_upstage import parse_documents
except Exception:
    parse_documents = None

# tool2 (검증)
try:
    from tools.tool2 import verify_ctd
except Exception:
    verify_ctd = None

# checklist validator
try:
    from validators.checklist_validate import validate_checklist_yaml
except Exception:
    validate_checklist_yaml = None

# tool3 (생성/저장)
try:
    from tools.generate_ctd import generate_ctd, save_output
except Exception:
    generate_ctd = None
    save_output = None


def _as_callable(fn_or_tool: Any) -> Optional[Callable]:
    """LangChain Tool이면 invoke로 감싸 일반 함수처럼 호출(래핑)."""
    if fn_or_tool is None:
        return None
    if hasattr(fn_or_tool, "invoke"):
        return lambda args=None, state=None, _tool=fn_or_tool: _tool.invoke(args or {})
    return fn_or_tool

TOOLS: Dict[str, Any] = {}
if parse_documents:
    TOOLS["parse_documents"] = _as_callable(parse_documents)
if verify_ctd:
    TOOLS["verify"] = _as_callable(verify_ctd)

if validate_checklist_yaml:
    def _checklist_validate_tool(args=None, state=None):
        path = None
        if isinstance(args, dict):
            path = args.get("yaml_or_path")
        path = path or os.getenv("CHECKLIST_PATH", "checklist.yaml")
        ok, errs = validate_checklist_yaml(path)
        return {"ok": bool(ok), "errors": list(errs), "path": path}
    TOOLS["checklist_validate"] = _checklist_validate_tool

if generate_ctd:
    TOOLS["generate_ctd"] = _as_callable(generate_ctd)
if save_output:
    TOOLS["save_output"] = _as_callable(save_output)

TOOL_SPEC = {
    "parse_documents":    {"args": {"file_paths": "List[str]", "texts": "List[str]"}, "desc": "파일/텍스트 파싱"},
    "verify":             {"args": {}, "desc": "문서 검증(RAG/규칙)"},
    "checklist_validate": {"args": {"yaml_or_path": "str?"}, "desc": "체크리스트 YAML 검증"},
    "generate_ctd":       {"args": {}, "desc": "최종 문서 생성"},
    "save_output":        {"args": {}, "desc": "결과 저장"},
}
