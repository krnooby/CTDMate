# ctd_agent_llama_chroma_5docs_upstage.py
# RAG(지식증강검색) : Chroma(벡터DB), 임베딩(embeddings: 문장 벡터화) : FastEmbed
# LLM : meta-llama/Llama-3.2-1B-Instruct + llama.cpp(온디바이스 추론 엔진)
# 파서(parser) : Upstage Document Parse API 사용(혼합형 PDF 지원)
# 흐름: 직렬 폴백(parse → verify → generate)

import os, uuid, json, re, time
from typing import TypedDict, Optional, List, Dict

from dotenv import load_dotenv

# LLM (llama.cpp, 온디바이스 추론 엔진)
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage

# RAG: Chroma(벡터DB 스토어) + FastEmbed(임베딩: 문장 벡터화)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from sentence_transformers import SentenceTransformer
import chromadb
import torch

# Upstage Document Parse(문서 파서)
from langchain_upstage import UpstageDocumentParseLoader

# 로컬 폴백 파서(optional)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

load_dotenv()

# ===== LLM 설정 =====
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/models/Llama-3.2-1B-Instruct.Q4_K_M.gguf")
LLAMA_CTX        = int(os.getenv("LLAMA_CTX", "4096"))
LLAMA_THREADS    = int(os.getenv("LLAMA_THREADS", str(os.cpu_count() or 8)))
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "800"))

# ===== Chroma 설정 =====
USE_CHROMA              = bool(int(os.getenv("USE_CHROMA", "1")))
CHROMA_DIR              = os.getenv("CHROMA_DIR", "./chroma_db")              # persistent directory(영구 저장 디렉터리)
CHROMA_DOC_COLLECTION   = os.getenv("CHROMA_DOC_COLLECTION", "ctd_verify_v1")  # 본문 임시 인덱스
CHROMA_REF_COLLECTION   = os.getenv("CHROMA_REF_COLLECTION", "")               # 참조 지식 인덱스(선택)
EMBED_MODEL             = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
REF_EMBED_MODEL         = os.getenv("REF_EMBED_MODEL", EMBED_MODEL)

# 점수 기준: similarity_search_with_score는 보통 distance(거리). 코사인 거리(cosine distance) → 유사도 = 1 - distance
SIM_THRESHOLD           = float(os.getenv("SIM_THRESHOLD", "0.78"))  # 유사도 임계값
K_TOP                   = int(os.getenv("K_TOP", "3"))

MAX_FILES   = int(os.getenv("MAX_FILES", "5"))

# ===== Upstage 설정 =====
UPSTAGE_API_KEY        = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_SPLIT          = os.getenv("UPSTAGE_SPLIT", "page")           # page | section
UPSTAGE_OUTPUT_FORMAT  = os.getenv("UPSTAGE_OUTPUT_FORMAT", "markdown") # markdown | html | text | json
DEMO_MODE              = bool(int(os.getenv("DEMO_MODE", "0")))        # 1이면 데모 더미 사용

REQUIRED_SECTIONS = [
    "효능 효과", "성분 및 함량", "용법 용량",
    "사용상의 주의사항", "약동학", "임상 시험 결과", "보관 및 취급 주의사항"
]

# ===== 상태 =====
class State(TypedDict, total=False):
    file_paths: Optional[List[str]]
    texts: Optional[List[str]]
    parsed_docs: List[Dict]
    combined_markdown: str
    verified: Dict
    output: str
    started_at: float

# ===== 유틸 =====

def _read_pdf_text_local(fp: str) -> str:
    """로컬 폴백 파서(PyMuPDF/pypdf). Upstage 실패 시 사용."""
    if fitz is not None:
        try:
            doc = fitz.open(fp)
            pages = [p.get_text("text") for p in doc]
            doc.close()
            return "\n".join(pages)
        except Exception:
            pass
    if PdfReader is not None:
        try:
            reader = PdfReader(fp)
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            pass
    return ""


def _combine_docs(items: List[Dict]) -> str:
    parts = []
    for i, it in enumerate(items, 1):
        parts.append(f"\n\n---\n### [DOC {i}] source: {it.get('source')}\n\n{it.get('markdown','')}\n")
    return "\n".join(parts).strip()


def _chunk(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def _embeddings():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)

_ST_CACHED = None

def _st_model():
    """SentenceTransformer 로더(참조 컬렉션 질의용). REF_EMBED_MODEL 사용."""
    global _ST_CACHED
    if _ST_CACHED is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ST_CACHED = SentenceTransformer(REF_EMBED_MODEL, device=device)
    return _ST_CACHED


def _chroma(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # 코사인 공간
    )


def _upstage_parse_file(fp: str) -> str:
    """Upstage Document Parse를 사용해 파일을 Markdown으로 파싱."""
    if DEMO_MODE:
        # 간단 더미
        base = [
            "# 의약품 설명서",
            "## 성분 및 함량\n- 성분 A 10mg\n- 성분 B 5mg",
            "## 용법 용량\n- 성인: 1일 2회 1정",
            "## 사용상의 주의사항\n- 임부 금기",
            "## 보관 및 취급 주의사항\n- 실온 보관",
        ]
        return "\n\n".join(base)

    if not UPSTAGE_API_KEY:
        # API 키 없으면 로컬 폴백
        return _read_pdf_text_local(fp)

    try:
        loader = UpstageDocumentParseLoader(
            fp,
            split=UPSTAGE_SPLIT,
            output_format=UPSTAGE_OUTPUT_FORMAT,
            api_key=UPSTAGE_API_KEY,
        )
        docs = loader.load()
        # output_format이 markdown/text가 아닐 수 있으므로 문자열화
        if UPSTAGE_OUTPUT_FORMAT in ("markdown", "text"):
            return "\n\n".join(d.page_content for d in docs)
        else:
            # HTML/JSON 등의 경우 텍스트로 단순 변환
            return "\n\n".join(str(d.page_content) for d in docs)
    except Exception:
        # 실패 시 로컬 폴백
        return _read_pdf_text_local(fp)


# ===== 1) parse =====

def parse_node(state: State) -> State:
    items: List[Dict] = []
    fps = state.get("file_paths") or []
    txts = state.get("texts") or []

    if fps:
        if len(fps) > MAX_FILES:
            fps = fps[:MAX_FILES]
        for fp in fps:
            md = _upstage_parse_file(fp)
            items.append({"source": fp, "markdown": md, "meta": {"parser": "upstage", "split": UPSTAGE_SPLIT, "fmt": UPSTAGE_OUTPUT_FORMAT}})

    if txts:
        if len(txts) > MAX_FILES:
            txts = txts[:MAX_FILES]
        for i, t in enumerate(txts, 1):
            items.append({"source": f"inline_{i}", "markdown": t, "meta": {"mode": "inline"}})

    combined = _combine_docs(items) if items else ""
    return {**state, "parsed_docs": items, "combined_markdown": combined, "started_at": state.get("started_at", time.time())}


# ===== 2) verify (Chroma RAG) =====

def verify_node(state: State) -> State:
    content = (state.get("combined_markdown") or "").strip()
    if not content or not USE_CHROMA:
        missing = [sec for sec in REQUIRED_SECTIONS if sec not in content]
        hits = {sec: ([] if sec in missing else [{"similarity": 0.99, "text": "섹션 발견"}]) for sec in REQUIRED_SECTIONS}
        return {**state, "verified": {"status": ("complete" if not missing else "incomplete"), "missing_fields": missing, "hits": hits, "ref_hits": {}}}

    # 인덱스 적재
    vs = _chroma(CHROMA_DOC_COLLECTION)
    run_id = str(uuid.uuid4())  # 충돌 방지용 ID prefix
    chunks = _chunk(content)
    if chunks:
        ids = [f"{run_id}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": "combined", "chunk_id": i} for i in range(len(chunks))]
        # 주의: 같은 컬렉션을 계속 쓰면 데이터가 누적됨. 운영 시 세션별 컬렉션을 권장.
        vs.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    hits, missing = {}, []
    for sec in REQUIRED_SECTIONS:
        pairs = vs.similarity_search_with_score(sec, k=K_TOP)  # score=distance(거리)
        rows = []
        top_sim = 0.0
        for doc, dist in pairs:
            sim = 1.0 - float(dist)
            rows.append({"similarity": round(sim, 3), "text": doc.page_content})
            top_sim = max(top_sim, sim)
        hits[sec] = rows
        if top_sim < SIM_THRESHOLD:
            missing.append(sec)

    # 참조 RAG(선택) — 기존 LangChain 래퍼 대신, 미리 임베딩된 컬렉션을 raw Chroma로 질의
    ref_hits = {}
    if CHROMA_REF_COLLECTION:
        try:
            try:
                from chromadb import PersistentClient
                ref_client = PersistentClient(path=CHROMA_DIR)
            except Exception:
                from chromadb.config import Settings
                ref_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
            ref_col = ref_client.get_or_create_collection(name=CHROMA_REF_COLLECTION)
            st = _st_model()
            for sec in missing:
                q = f"{sec} 작성 가이드"
                q_vec = st.encode([q], convert_to_numpy=True).tolist()
                qr = ref_col.query(query_embeddings=q_vec, n_results=K_TOP, include=["documents","metadatas","distances"])
                rows = []
                docs = (qr.get("documents") or [[]])[0]
                metas = (qr.get("metadatas") or [[]])[0]
                dists = (qr.get("distances") or [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    sim = 1.0 - float(dist) if dist is not None else 0.0
                    rows.append({"similarity": round(sim, 3), "text": str(doc)[:800], "source": (meta or {}).get("source", "")})
                ref_hits[sec] = rows
        except Exception:
            pass

    status = "complete" if not missing else "incomplete"
    return {**state, "verified": {"status": status, "missing_fields": missing, "hits": hits, "ref_hits": ref_hits}}


# ===== 3) generate =====

def generate_node(state: State) -> State:
    ctx = state.get("combined_markdown", "")
    vf = state.get("verified", {})
    notes = []
    for sec, rows in (vf.get("ref_hits") or {}).items():
        if rows:
            notes.append(f"- {sec}: {rows[0]['text'][:200].replace(chr(10),' ')}")

    prompt = f"""당신은 의약품 CTD 문서 생성기입니다.
아래 정보를 바탕으로 한국어 Markdown으로 CTD 요약을 작성하세요.
누락된 항목: {', '.join(vf.get('missing_fields', [])) or '없음'}

[입력]
{ctx[:6000]}

[참조 RAG 힌트]
{os.linesep.join(notes) if notes else '없음'}

[출력 형식]
# CTD 요약
## 효능 효과
## 성분 및 함량
## 용법 용량
## 사용상의 주의사항
## 약동학
## 임상 시험 결과
## 보관 및 취급 주의사항
"""
    try:
        llm = ChatLlamaCpp(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_CTX,
            n_threads=LLAMA_THREADS,
            temperature=0.1,
            max_tokens=LLAMA_MAX_TOKENS,
            verbose=False,
        )
        msg = llm.invoke([HumanMessage(content=prompt)])
        out = getattr(msg, "content", str(msg))
    except Exception as e:
        out = f"# CTD 요약\n- LLM 생성 실패 폴백: {e}\n"
    return {**state, "output": out}


# ===== 직렬 폴백 실행 =====

def run_pipeline(file_paths: Optional[List[str]] = None, texts: Optional[List[str]] = None) -> Dict:
    st: State = {"file_paths": file_paths, "texts": texts, "started_at": time.time()}
    st = parse_node(st)
    st = verify_node(st)
    st = generate_node(st)
    return st


# ===== 실행 예시 =====
if __name__ == "__main__":
    demos = [
        {"file_paths": ["./a.pdf", "./b.pdf", "./c.pdf"], "texts": None},
        {"file_paths": None, "texts": ["문서1 텍스트", "문서2 텍스트"]},
        {"file_paths": ["./d1.pdf", "./d2.pdf", "./d3.pdf", "./d4.pdf", "./d5.pdf", "./d6.pdf"], "texts": None},
    ]
    for i, ex in enumerate(demos, 1):
        print(f"\n========== [데모 {i}] ==========")
        out = run_pipeline(**ex)
        print(out.get("output", ""))
