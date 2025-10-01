"""
RAG Infrastructure Module

이 모듈은 CTDMate의 RAG(Retrieval-Augmented Generation) 인프라를 제공합니다.

주요 구성요소:
- glossary_rag.py: 용어집 검색
- mfds_rag.py: MFDS 가이드라인 검색
- CTD_rag.py: CTD 템플릿 검색
- retriever.py: Hybrid 검색 (Vector+BM25+MMR)
- indexer.py: 문서 인덱싱 (Qdrant)
- term_normalizer.py: 용어 정규화 (Llama-3.2)
- bm25_search.py: BM25 구현
- chunking.py: 문서 청킹 유틸
- embedding.py: 임베딩 유틸
- regulation_*.py: 규제 검증 상태/노드/엣지
"""

from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

__all__ = [
    "PROJECT_ROOT",
]
