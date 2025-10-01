# ctdmate/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

# config
try:
    from ctdmate.app import config as CFG
except Exception:
    from .app import config as CFG  # type: ignore

# types
try:
    from ctdmate.app.types import ParseOutput, ValidateExcelOutput, GenerateOutput, RoutePlan
except Exception:
    from .app.types import ParseOutput, ValidateExcelOutput, GenerateOutput, RoutePlan  # type: ignore

# brain
try:
    from ctdmate.brain.router import Router, LlamaLocalClient
except Exception:
    from .brain.router import Router, LlamaLocalClient  # type: ignore

# tools
try:
    from ctdmate.tools.smartdoc_upstage import run as parse_run
    from ctdmate.tools.reg_rag import RegulationRAGTool
    from ctdmate.tools.gen_solar import SolarGenerator
except Exception:
    from .tools.smartdoc_upstage import run as parse_run  # type: ignore
    from .tools.reg_rag import RegulationRAGTool  # type: ignore
    from .tools.gen_solar import SolarGenerator  # type: ignore


class CTDPipeline:
    """
    Router → Parse → Validate → Generate
    - Router: Llama3.2-3B가 action/section/format 결정
    - Parse: Upstage Document Parse → Markdown/JSONL
    - Validate: 규제 커버리지·신뢰도·위반 스코어 계산
    - Generate: Solar Pro2 + 인용형 RAG, Lint 게이트
    """

    def __init__(self, llama_client: Optional[LlamaLocalClient] = None):
        self.router = Router(llama=llama_client)
        self.reg_tool = RegulationRAGTool(
            auto_normalize=True,
            enable_rag=True,
            llama_client=llama_client,
        )
        self.gen = SolarGenerator(
            enable_rag=True,
            auto_normalize=True,
            output_format="yaml",
        )

    def execute(
        self,
        user_desc: str,
        files: Optional[List[str]] = None,
        section: Optional[str] = None,
        output_format: Optional[str] = None,
        auto_fix: bool = True,
    ) -> Dict[str, Any]:
        plan: RoutePlan = self.router.decide(user_desc)
        if section:
            plan["section"] = section
        if output_format:
            plan["output_format"] = output_format

        # Parse
        parse_out: Optional[ParseOutput] = None
        if plan.get("need_parse") and files:
            parse_out = parse_run(files)

        # Validate
        validate_out: Optional[Dict[str, Any]] = None
        content_for_validate = user_desc
        # 엑셀 시트 우선 검증
        if parse_out and parse_out.get("results"):
            for r in parse_out["results"]:
                if str(r["input"]).lower().endswith(".xlsx"):
                    validate_out = self.reg_tool.validate_excel(r["input"], auto_fix=auto_fix)
                    break
        # 엑셀이 없으면 텍스트 단일 검증
        if plan.get("need_validate") and validate_out is None:
            validate_out = self.reg_tool.validate_and_normalize(
                section=plan.get("section") or "M2.3",
                content=content_for_validate,
                auto_fix=auto_fix,
            )

        # Decide gate for generation
        ok_for_gen = True
        normalized = user_desc
        if isinstance(validate_out, dict) and "metrics" in validate_out:
            ok_for_gen = bool(validate_out["pass"]) and (
                float(validate_out["metrics"]["score"]) >= CFG.GENERATE_GATE
            )
            normalized = validate_out.get("normalized_content") or user_desc

        # Generate
        generate_out: Optional[GenerateOutput] = None
        if plan.get("need_generate"):
            if ok_for_gen:
                generate_out = self.gen.generate(
                    section=plan.get("section") or "M2.3",
                    prompt=normalized,
                    output_format=plan.get("output_format") or "yaml",
                )
            else:
                generate_out = {  # type: ignore[assignment]
                    "section": plan.get("section") or "M2.3",
                    "format": plan.get("output_format") or "yaml",
                    "text": "",
                    "rag_used": False,
                    "rag_refs": [],
                    "lint_ok": False,
                    "lint_findings": [],
                    "gen_metrics": {"gen_score": 0.0},
                    "ready": False,
                    "offline_fallback": None,
                }

        return {
            "plan": plan,
            "parse": parse_out,
            "validate": validate_out,
            "generate": generate_out,
        }


# CLI
def _read_text(p: Optional[str]) -> str:
    if not p:
        return ""
    path = Path(p)
    return path.read_text(encoding="utf-8") if path.exists() else p


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="CTDMate pipeline: Router→Parse→Validate→Generate")
    ap.add_argument("--desc", "-d", required=True, help="요청 설명 또는 프롬프트 텍스트/파일 경로")
    ap.add_argument("--files", "-f", nargs="*", help="파싱 대상 파일 목록(.pdf/.xlsx)")
    ap.add_argument("--section", "-s", help="강제 섹션(예: M2.3, M2.6, M2.7)")
    ap.add_argument("--format", "-o", choices=["yaml", "markdown"], help="출력 형식")
    ap.add_argument("--no-autofix", action="store_true", help="자동 정규화 비활성")
    args = ap.parse_args()

    desc = _read_text(args.desc)
    pipe = CTDPipeline()
    out = pipe.execute(
        user_desc=desc,
        files=args.files or [],
        section=args.section,
        output_format=args.format,
        auto_fix=not args.no_autofix,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
