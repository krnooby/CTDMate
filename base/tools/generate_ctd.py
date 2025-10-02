# tools/generate_ctd.py
import os, json, re, time, textwrap, hashlib
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool                     # Tool(도구) 데코레이터
from langchain_upstage import ChatUpstage                 # Upstage LLM 클라이언트(모델 호출기)

UPSTAGE_API_KEY      = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_MODEL        = os.getenv("UPSTAGE_GEN_MODEL", "solar-pro2")
UPSTAGE_TEMPERATURE  = float(os.getenv("UPSTAGE_TEMPERATURE", "0.1"))
UPSTAGE_MAX_TOKENS   = int(os.getenv("UPSTAGE_MAX_TOKENS", "1800"))
OUTPUT_DIR           = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()

REQUIRED_SECTIONS = [
    "제1부 신청내용 및 행정정보",
    "제2부 국제공통기술문서의 자료개요 및 요약",
    "2.1 목차",
    "2.2 서론",
    "2.3 품질평가자료요약",
    "2.4 비임상시험자료개요",
    "2.5 임상시험자료개요",
    "2.6 비임상시험자료요약문 및 요약표",
    "2.7 임상시험자료요약",
    "제3부 품질평가자료",
    "제4부 비임상시험자료",
    "제5부 임상시험자료",
]

def _load_from_documents_json(documents_json: Optional[str]) -> str:
    if not documents_json:
        return ""
    try:
        obj = json.loads(documents_json)
        return obj.get("combined_markdown") or ""
    except Exception:
        return ""

def _load_missing_from_verified(verified_json: Optional[str]) -> List[str]:
    if not verified_json:
        return []
    try:
        obj = json.loads(verified_json)
        return list(obj.get("missing_fields") or [])
    except Exception:
        return []

def _ellipsis(s: str, n: int) -> str:
    s = s or ""
    return s[:n]

def _slug(s: str, fallback: str = "output") -> str:
    s = (s or "").strip()
    if not s:
        return fallback
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or fallback

def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]

@tool("generate_ctd")
def generate_ctd(
    documents_json: Optional[str] = None,   # parse 결과(JSON 문자열)
    verified_json: Optional[str] = None,    # verify 결과(JSON 문자열)
    normalize_json: Optional[str] = None    # 정규화 규칙(JSON) — 실제 적용은 상위 로직에서 수행
) -> str:
    """
    Tool3: Upstage Solar API로 CTD 최종 문서 생성(generation, 생성).
    반환(JSON 문자열): {"markdown": "...", "meta": {...}}
    """
    ctx_markdown = _load_from_documents_json(documents_json)
    missing = _load_missing_from_verified(verified_json)

    prompt = f"""당신은 의약품 CTD 작성 전문가입니다.
다음 입력을 바탕으로 한국어 Markdown(마크다운, 경량 문서 포맷)으로 **요청된 CTD 구조**에 맞게 완결형 문서를 작성하세요.
전문용어는 최초 1회 괄호에 한글 설명을 병기하세요(예: pharmacodynamics(약력학)).

[참고 힌트]
- 필수 섹션: {", ".join(REQUIRED_SECTIONS)}
- 누락 섹션 추정: {", ".join(missing) if missing else "없음"}

[본문 합본]
{_ellipsis(ctx_markdown, 6000)}

[정규화 규칙(JSON)]  # 실제 적용은 상위 rules.py에서 수행
{_ellipsis(normalize_json or "", 3000)}

[출력 형식(이 구조와 헤딩 레벨을 그대로 사용)]
# CTD의 구성
## 제1부 신청내용 및 행정정보

## 제2부 국제공통기술문서의 자료개요 및 요약
### 2.1 목차
### 2.2 서론
### 2.3 품질평가자료요약
### 2.4 비임상시험자료개요
### 2.5 임상시험자료개요
### 2.6 비임상시험자료요약문 및 요약표
### 2.7 임상시험자료요약

## 제3부 품질평가자료

## 제4부 비임상시험자료

## 제5부 임상시험자료
"""

    if not UPSTAGE_API_KEY:
        return json.dumps(
            {"markdown": "# CTD의 구성\n- 오류: UPSTAGE_API_KEY 누락.",
             "meta": {"ok": False}},
            ensure_ascii=False
        )

    try:
        llm = ChatUpstage(
            api_key=UPSTAGE_API_KEY,
            model=UPSTAGE_MODEL,
            temperature=UPSTAGE_TEMPERATURE,
            max_tokens=UPSTAGE_MAX_TOKENS,
        )
        msg = llm.invoke(prompt)
        text = getattr(msg, "content", "").strip()
        if not text:
            raise RuntimeError("empty_response")
        return json.dumps({
            "markdown": text,
            "meta": {
                "ok": True,
                "model": UPSTAGE_MODEL,
                "missing_sections_from_verify": missing,
                "used_inputs": {
                    "documents_json": bool(documents_json),
                    "verified_json": bool(verified_json),
                    "normalize_json": bool(normalize_json)
                }
            }
        }, ensure_ascii=False)
    except Exception as e:
        try:
            llm = ChatUpstage(
                api_key=UPSTAGE_API_KEY,
                model="solar-pro",
                temperature=UPSTAGE_TEMPERATURE,
                max_tokens=UPSTAGE_MAX_TOKENS,
            )
            msg = llm.invoke(prompt)
            text = getattr(msg, "content", "").strip()
            return json.dumps({
                "markdown": text or "# CTD의 구성\n- 생성 실패 폴백.",
                "meta": {"ok": text != "", "model": "solar-pro", "error": str(e)}
            }, ensure_ascii=False)
        except Exception as e2:
            return json.dumps({
                "markdown": "# CTD의 구성\n- 생성 실패: " + str(e2),
                "meta": {"ok": False, "error": str(e)}
            }, ensure_ascii=False)

def save_output(
    markdown: Optional[str] = None,            # 직접 마크다운 입력(마크다운)
    result_json: Optional[str] = None,         # generate_ctd 결과(JSON 문자열). 있으면 여기서 markdown 추출
    output_dir: Optional[str] = None,          # 출력 디렉터리(출력 폴더)
    filename_stem: Optional[str] = None        # 파일명 접두(파일명 앞부분)
) -> str:
    """
    마크다운을 PDF로 저장(PDF 내보내기).
    1) markdown→HTML→PDF(WeasyPrint, HTML→PDF 변환기) 시도
    2) ReportLab(리포트랩, PDF 생성기)로 텍스트 PDF 생성
    3) 둘 다 없으면 예외
    반환: 생성된 PDF 파일 경로 문자열
    """
    if not markdown and result_json:
        try:
            obj = json.loads(result_json)
            markdown = obj.get("markdown") or ""
        except Exception:
            markdown = markdown or ""

    if not markdown:
        raise ValueError("markdown 또는 result_json 중 하나는 필요합니다.")

    out_dir = Path(output_dir).resolve() if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = filename_stem or f"ctd-{time.strftime('%Y%m%d-%H%M%S')}-{_hash(markdown)}"
    stem = _slug(stem)
    pdf_path = out_dir / f"{stem}.pdf"

    # A) WeasyPrint 경로(HTML→PDF)
    html_text = None
    try:
        import markdown as md  # 파이썬-Markdown(마크다운→HTML 변환기)
        html_text = md.markdown(markdown or "", extensions=["tables", "fenced_code"])
        try:
            from weasyprint import HTML  # WeasyPrint(HTML→PDF 렌더러)
            HTML(string=html_text).write_pdf(str(pdf_path))
            return str(pdf_path)
        except Exception:
            html_text = None  # 다음 경로로 폴백
    except Exception:
        pass

    # B) ReportLab 경로(텍스트→PDF)
    try:
        from reportlab.pdfgen import canvas         # ReportLab(리포트랩, PDF 생성기)
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # 한글 폰트 등록 시도: 환경변수 PDF_TTF 경로 우선
        font_path = os.getenv("PDF_TTF", "")
        font_name = "Helvetica"  # 기본 폰트(라틴) — 한글 미출력 가능
        if font_path and Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("UserTTF", font_path))
                font_name = "UserTTF"
            except Exception:
                pass

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        left_margin, top_margin = 50, height - 50
        line_height = 14

        c.setFont(font_name, 11)
        text_obj = c.beginText(left_margin, top_margin)

        def _md_to_lines(md_src: str):
            raw = (md_src or "").replace("\r\n", "\n").replace("\r", "\n")
            lines = []
            for ln in raw.split("\n"):
                ln = re.sub(r"^\s{0,3}(#{1,6}\s*)", "", ln)  # 헤딩 제거(제목)
                ln = re.sub(r"\*\*([^*]+)\*\*", r"\1", ln)  # 굵게 제거(볼드)
                ln = re.sub(r"\*([^*]+)\*", r"\1", ln)      # 기울임 제거(이탤릭)
                ln = re.sub(r"`([^`]+)`", r"\1", ln)        # 인라인 코드 제거
                lines.extend(textwrap.wrap(ln, width=90) or [""])
            return lines

        for line in _md_to_lines(markdown or ""):
            text_obj.textLine(line)
            if text_obj.getY() <= 50:
                c.drawText(text_obj); c.showPage()
                c.setFont(font_name, 11)
                text_obj = c.beginText(left_margin, top_margin)

        c.drawText(text_obj)
        c.save()
        return str(pdf_path)
    except Exception as e:
        raise RuntimeError(
            "PDF 생성 실패. weasyprint 또는 reportlab 설치 필요."
        ) from e
