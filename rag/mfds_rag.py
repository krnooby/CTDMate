"""
MFDS RAG Tool

Qdrant Vector DB에 인덱싱된 MFDS/ICH 규제 가이드라인을 검색하는 RAG 도구입니다.
- CTD 작성 가이드라인 검색
- 모듈/섹션별 필터링
- 체크리스트 및 교차참조 정보 제공

사용 예시:
    from tools.mfds_rag import MFDSRAGTool

    rag = MFDSRAGTool()
    results = rag.search_guideline("M2.7 임상 요약 작성법", k=5)
    results = rag.search_by_module("임상약리", module="M2.7", k=3)
"""

import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.embedding import E5Embedder


class E5EmbeddingAdapter(Embeddings):
    """
    LangChain의 Embeddings 인터페이스에 맞춰 E5Embedder를 래핑하는 어댑터
    """
    def __init__(self, embedder: E5Embedder):
        super().__init__()
        self.embedder = embedder
        self.task_description = "주어진 CTD 관련 질문에 대해, 규제 가이드라인에서 가장 관련성 높은 섹션을 검색하시오"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 임베딩 (배치 처리)"""
        embeddings = self.embedder.embed_documents(texts, batch_size=32, show_progress_bar=False)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        """쿼리 임베딩 (instruction 포함)"""
        embeddings = self.embedder.embed_queries([text], task=self.task_description, batch_size=1)
        return embeddings.cpu().numpy().tolist()[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """비동기 문서 임베딩 (동기 메서드 호출)"""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """비동기 쿼리 임베딩 (동기 메서드 호출)"""
        return self.embed_query(text)


class MFDSRAGTool:
    """
    Qdrant 기반 MFDS 가이드라인 검색 도구

    Attributes:
        vector_store: QdrantVectorStore 인스턴스
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        collection_name: str = "mfds_guidelines",
        model_name: str = "intfloat/multilingual-e5-large-instruct"
    ):
        """
        MFDSRAGTool 초기화

        Args:
            storage_path: Qdrant 저장 경로 (기본: ./qdrant_storage/mfds)
            collection_name: 검색할 컬렉션 이름
            model_name: 임베딩 모델 이름
        """
        if storage_path is None:
            storage_path = str(project_root / "qdrant_storage" / "mfds")

        self.storage_path = storage_path
        self.collection_name = collection_name

        # Qdrant 클라이언트 초기화
        self.client = QdrantClient(path=storage_path)

        # E5 임베딩 모델 초기화
        print(f"🤖 E5 임베딩 모델 로딩 중... ({model_name})")
        embedder = E5Embedder(model_name=model_name)
        self.embedding_adapter = E5EmbeddingAdapter(embedder)

        # Vector Store 초기화
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding_adapter
        )

        print(f"✅ MFDSRAGTool 초기화 완료")
        print(f"   - Storage: {storage_path}")
        print(f"   - Collection: {collection_name}\n")

    def search_guideline(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        가이드라인 검색 (유사도 기반)

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 최소 유사도 점수 (0~1)

        Returns:
            검색 결과 리스트 [{"module", "section", "title", "description", "checklist", "cross_ref", "score", "content"}]
        """
        print(f"🔍 검색 쿼리: '{query}' (top-{k})")

        if score_threshold:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
            )
            # 점수 필터링
            results = [(doc, score) for doc, score in results if score >= score_threshold]
        else:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

        # 결과 포맷팅
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "module": doc.metadata.get("module", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "description": doc.metadata.get("description", "N/A"),
                "checklist": doc.metadata.get("checklist", []),
                "cross_ref": doc.metadata.get("cross_ref", []),
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        print(f"📊 검색 완료: {len(formatted_results)}개 결과 반환\n")
        return formatted_results

    def search_by_module(
        self,
        query: str,
        module: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        모듈별 필터링 검색

        Args:
            query: 검색 쿼리
            module: 필터링할 모듈 (예: "M2.7", "M2.6")
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        print(f"🔍 모듈 필터링 검색: '{query}' [모듈: {module}]")

        # Qdrant 필터 생성 (중첩 구조 접근)
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.module",
                    match=MatchValue(value=module)
                )
            ]
        )

        # 검색 실행
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_condition
        )

        # 결과 포맷팅
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "module": doc.metadata.get("module", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "description": doc.metadata.get("description", "N/A"),
                "checklist": doc.metadata.get("checklist", []),
                "cross_ref": doc.metadata.get("cross_ref", []),
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        print(f"📊 검색 완료: {len(formatted_results)}개 결과 반환\n")
        return formatted_results

    def search_with_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        MMR (Maximal Marginal Relevance) 기반 다양성 검색

        Args:
            query: 검색 쿼리
            k: 최종 반환 결과 수
            fetch_k: 초기 검색 결과 수
            lambda_mult: 관련성 vs 다양성 가중치 (0~1, 높을수록 관련성 우선)

        Returns:
            검색 결과 리스트
        """
        print(f"🔍 MMR 검색: '{query}' (k={k}, fetch_k={fetch_k}, λ={lambda_mult})")

        results = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

        # 결과 포맷팅
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "module": doc.metadata.get("module", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "description": doc.metadata.get("description", "N/A"),
                "checklist": doc.metadata.get("checklist", []),
                "cross_ref": doc.metadata.get("cross_ref", []),
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        print(f"📊 MMR 검색 완료: {len(formatted_results)}개 결과 반환\n")
        return formatted_results

    def get_section_by_id(self, section_id: str) -> Optional[Dict[str, Any]]:
        """
        섹션 ID로 정확히 검색

        Args:
            section_id: 섹션 ID (예: "2.7", "2.6.2")

        Returns:
            검색 결과 (없으면 None)
        """
        print(f"🎯 섹션 ID 검색: '{section_id}'")

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.section",
                    match=MatchValue(value=section_id)
                )
            ]
        )

        results = self.vector_store.similarity_search(
            query=section_id,
            k=1,
            filter=filter_condition
        )

        if results:
            doc = results[0]
            return {
                "module": doc.metadata.get("module", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "description": doc.metadata.get("description", "N/A"),
                "checklist": doc.metadata.get("checklist", []),
                "cross_ref": doc.metadata.get("cross_ref", []),
                "content": doc.page_content,
                "metadata": doc.metadata
            }

        print(f"❌ 섹션을 찾을 수 없습니다: '{section_id}'\n")
        return None

    @staticmethod
    def parse_json_content(content: str) -> Dict[str, Any]:
        """
        검색 결과의 content(JSON 문자열)를 파싱하여 딕셔너리로 반환

        Args:
            content: JSON 문자열 (검색 결과의 'content' 필드)

        Returns:
            파싱된 가이드라인 데이터 딕셔너리
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 파싱 오류: {e}")
            return {}

    def get_checklist(self, section_id: str) -> Optional[List[str]]:
        """
        특정 섹션의 체크리스트 반환

        Args:
            section_id: 섹션 ID

        Returns:
            체크리스트 (없으면 None)
        """
        result = self.get_section_by_id(section_id)
        if result:
            return result['checklist']
        return None


# 사용 예시 및 테스트
if __name__ == "__main__":
    print("=" * 70)
    print("  MFDS RAG Tool 테스트")
    print("=" * 70)
    print()

    # 도구 초기화
    rag = MFDSRAGTool()

    # 테스트 1: 일반 검색
    print("\n" + "=" * 70)
    print("  테스트 1: 일반 검색")
    print("=" * 70 + "\n")

    results = rag.search_guideline("임상 요약 작성 방법", k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['section']}] {result['title']} (유사도: {result['score']:.4f})")
        print(f"   Module: {result['module']}")
        print(f"   설명: {result['description'][:100]}...")
        print(f"   체크리스트 항목 수: {len(result['checklist'])}")
        print()

    # 테스트 2: 모듈 필터링
    print("\n" + "=" * 70)
    print("  테스트 2: 모듈 필터링 검색")
    print("=" * 70 + "\n")

    results = rag.search_by_module("약리", module="M2.6", k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['section']}] {result['title']} (유사도: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()

    # 테스트 3: 섹션 ID 검색
    print("\n" + "=" * 70)
    print("  테스트 3: 섹션 ID 직접 검색")
    print("=" * 70 + "\n")

    result = rag.get_section_by_id("2.7")
    if result:
        print(f"✅ 섹션 발견: [{result['section']}] {result['title']}")
        print(f"   설명: {result['description']}")
        print(f"\n   체크리스트 ({len(result['checklist'])}개):")
        for i, item in enumerate(result['checklist'][:3], 1):
            print(f"     {i}. {item}")
        print(f"\n   교차참조: {', '.join(result['cross_ref'])}")
    else:
        print("❌ 섹션을 찾을 수 없습니다.")

    # 테스트 4: 체크리스트 조회
    print("\n" + "=" * 70)
    print("  테스트 4: 체크리스트 조회")
    print("=" * 70 + "\n")

    checklist = rag.get_checklist("2.7")
    if checklist:
        print(f"✅ 섹션 2.7 체크리스트 ({len(checklist)}개 항목):")
        for i, item in enumerate(checklist, 1):
            print(f"  {i}. {item}")
    else:
        print("❌ 체크리스트를 찾을 수 없습니다.")

    print("\n" + "=" * 70)
    print("  테스트 완료")
    print("=" * 70)
