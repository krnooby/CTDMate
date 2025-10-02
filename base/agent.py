import os, re, json, logging
from typing import Any, Dict, List, Optional
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage
from registry import TOOLS, TOOL_SPEC
from settings import (
    LLAMA_MODEL_PATH, LLAMA_CTX, LLAMA_THREADS, LLAMA_MAX_TOKENS, LOG_LEVEL
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("ctd-react")

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
- parse_documents → verify → checklist_validate → generate_ctd → save_output
- 도구 외 임의의 실행 금지, JSON 키는 반드시 "tool","args"만 사용.
"""

def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"Action:\s*```json\s*(\{.*?\})\s*```", text, flags=re.S) or \
        re.search(r"Action:\s*(\{.*\})", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def _run_tool(tool_obj: Any, args: Dict[str, Any], state: Dict[str, Any]) -> Any:
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(args)
    try:
        return tool_obj(args, state)
    except TypeError:
        try:
            return tool_obj(**(args or {}))
        except TypeError:
            return tool_obj()

def run_agent(file_paths=None, texts=None, max_steps: int = 8) -> Dict[str, Any]:
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

        m_final = re.search(r"FinalAnswer:\s*(.+)", text, flags=re.S)
        if m_final and not _extract_tool_call(text):
            state["final_message"] = m_final.group(1).strip()
            break

        call = _extract_tool_call(text)
        if not call:
            next_hint = {"tool": "parse_documents", "args": user_hint} if step == 1 else {"tool": "generate_ctd", "args": {}}
            history.append(HumanMessage(content=f"Action: {json.dumps(next_hint, ensure_ascii=False)}"))
            continue

        tool_name = call.get("tool")
        args = call.get("args") or {}
        tool_obj = TOOLS.get(tool_name)
        if not tool_obj:
            history.append(HumanMessage(content=f"Observation: invalid tool '{tool_name}'. Use one of {list(TOOLS)}"))
            continue

        if tool_name == "parse_documents":
            args.setdefault("file_paths", user_hint["file_paths"])
            args.setdefault("texts", user_hint["texts"])

        try:
            result = _run_tool(tool_obj, args, state)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        history.append(HumanMessage(content=f"Observation: {json.dumps(result, ensure_ascii=False)[:2000]}"))
        if tool_name == "save_output" and isinstance(result, dict) and result.get("path"):
            history.append(HumanMessage(content=f"FinalAnswer: saved -> {result['path']}"))
            state["saved_path"] = result["path"]
            break
    return state
