# ctd_agent_react_min.py
# ReAct 에이전트(온디바이스 Llama가 툴 선택) + 외부 툴 연결만
import os, re, json, logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv; load_dotenv()
from base import run_agent
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage

# ---------- 로깅 ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("ctd-react-min")

# ---------- LLM(플래너) ----------
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/models/Llama-3.2-1B-Instruct.Q4_K_M.gguf")
LLAMA_CTX        = int(os.getenv("LLAMA_CTX", "4096"))
LLAMA_THREADS    = int(os.getenv("LLAMA_THREADS", str(os.cpu_count() or 8)))
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "800"))

# ---------- 툴 임포트(이미 구현되어 있다고 가정) ----------
# 필요에 맞게 경로만 맞춰주세요.
try:
    from tools.smartdoc_upstage import parse_documents
except Exception:
    parse_documents = None
try:
    from tools.tool2 import verify_ctd
except Exception:
    verify_ctd = None
try:
    from validators.checklist_validate import checklist_validate
except Exception:
    checklist_validate = None
try:
    from tools.generate_ctd import generate_ctd
except Exception:
    generate_ctd = None
try:
    from tools.generate_ctd import save_output
except Exception:
    save_output = None

VERIFY_TOOL_NAME = "verify_ctd" if verify_ctd else None

# 실제 등록되는 툴 레지스트리
TOOLS: Dict[str, Any] = {}
if parse_documents:     TOOLS["smartdoc_upstage"]     = parse_documents
if VERIFY_TOOL_NAME:    TOOLS["tool2"]                = verify_ctd
if checklist_validate:  TOOLS["checklist_validate"]  = checklist_validate
if generate_ctd:        TOOLS["generate_ctd"]        = generate_ctd
if save_output:         TOOLS["save_output"]         = save_output

# 툴 사양(LLM 안내용)
TOOL_SPEC = {
    "parse_documents":    {"args": {"file_paths":"List[str]","texts":"List[str]"}, "desc":"파일/텍스트 파싱"},
    (VERIFY_TOOL_NAME or "verify"): {"args": {}, "desc":"문서 검증(RAG/규칙 등)"},
    "checklist_validate": {"args": {}, "desc":"체크리스트/RAG 필터 검증"},
    "generate_ctd":       {"args": {}, "desc":"최종 문서 생성"},
    "save_output":        {"args": {}, "desc":"결과 저장"},
}
# 실제 없는 키는 표시만 되고 호출은 되지 않음(LLM이 선택시 에이전트가 무효 툴이라고 알려줌)

SYSTEM_PROMPT = f"""당신은 ReAct 스타일의 CTD 에이전트입니다.
다음 툴들만 사용하세요(반드시 JSON 형식으로 지시):
{json.dumps(TOOL_SPEC, ensure_ascii=False, indent=2)}

반드시 아래 형식을 따르세요:
Thought: (다음에 무엇을 할지 한 줄)
Action: {{ "tool": "<tool_name>", "args": {{ ... }} }}
(도구 응답을 받은 뒤) Observation: (간단 요약)
... 반복 ...
FinalAnswer: (최종 요약 또는 저장 결과 경로)

권장 순서:
- parse_documents → {VERIFY_TOOL_NAME or "verify"} → checklist_validate → generate_ctd → save_output
- 도구 외 임의의 실행 금지, JSON 키는 반드시 "tool","args"만 사용.
"""

def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    # ```json 블록 또는 Action: {..} 에서 JSON 추출
    m = re.search(r"Action:\s*```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if not m:
        m = re.search(r"Action:\s*(\{.*\})", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def _run_tool(tool_obj: Any, args: Dict[str, Any], state: Dict[str, Any]) -> Any:
    """
    툴 호출 어댑터:
      1) LangChain Tool: tool.invoke(args)
      2) 커스텀 함수(상태 사용): tool(args, state)
      3) 커스텀 함수(인자만): tool(**args)
      4) 인자 없는 함수: tool()
    """
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(args)
    try:
        return tool_obj(args, state)
    except TypeError:
        try:
            return tool_obj(**(args or {}))
        except TypeError:
            return tool_obj()

def run_agent(file_paths: Optional[List[str]] = None,
              texts: Optional[List[str]] = None,
              max_steps: int = 8) -> Dict[str, Any]:
    """
    단일 에이전트(LLM 플래너)가 ReAct로 툴을 호출.
    state는 툴들이 자유롭게 사용/갱신(예: documents_json 등)할 수 있음.
    """
    state: Dict[str, Any] = {}
    llama = ChatLlamaCpp(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=LLAMA_CTX,
        n_threads=LLAMA_THREADS,
        temperature=0.0,
        max_tokens=LLAMA_MAX_TOKENS,
        verbose=False,
    )

    user_hint = {"file_paths": file_paths or [], "texts": texts or []}
    history: List[HumanMessage] = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Inputs: {json.dumps(user_hint, ensure_ascii=False)}")
    ]

    for step in range(1, max_steps + 1):
        ai = llama.invoke(history)
        text = getattr(ai, "content", "")
        log.info("LLM step %d\n%s", step, text)

        # 끝?
        m_final = re.search(r"FinalAnswer:\s*(.+)", text, flags=re.S)
        if m_final and not _extract_tool_call(text):
            state["final_message"] = m_final.group(1).strip()
            break

        # 툴 호출 파싱
        call = _extract_tool_call(text)
        if not call:
            # 첫스텝에 액션이 없으면 parse_documents 힌트 제공
            next_hint = {"tool": "parse_documents", "args": user_hint} if step == 1 else {"tool": "generate_ctd", "args": {}}
            history.append(HumanMessage(content=f"Action: {json.dumps(next_hint, ensure_ascii=False)}"))
            continue

        tool_name = call.get("tool")
        args = call.get("args") or {}
        tool_obj = TOOLS.get(tool_name)

        if not tool_obj:
            history.append(HumanMessage(content=f"Observation: invalid tool '{tool_name}'. Use one of {list(TOOLS)}"))
            continue

        # parse_documents에는 초깃값 주입
        if tool_name == "parse_documents":
            args.setdefault("file_paths", user_hint["file_paths"])
            args.setdefault("texts", user_hint["texts"])

        try:
            result = _run_tool(tool_obj, args, state)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        obs = json.dumps(result, ensure_ascii=False)
        history.append(HumanMessage(content=f"Observation: {obs[:2000]}"))

        # save_output 성공 시 종료 힌트
        if tool_name == "save_output" and isinstance(result, dict) and result.get("path"):
            history.append(HumanMessage(content=f"FinalAnswer: saved -> {result['path']}"))
            state["saved_path"] = result["path"]
            break

    return state

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CTD ReAct agent (planner + external tools)")
    p.add_argument("--files","-f", nargs="*", help="input files")
    p.add_argument("--texts","-t", nargs="*", help="inline texts")
    args = p.parse_args()

    out = run_agent(file_paths=args.files or [], texts=args.texts or [])
    print(out.get("final_message",""))
    if out.get("saved_path"):
        print("Saved:", out["saved_path"])
