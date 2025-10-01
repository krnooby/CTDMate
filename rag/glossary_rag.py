"""
Glossary RAG Tool

Qdrant Vector DB에 인덱싱된 용어집을 검색하는 RAG 도구입니다.
- 의료/제약 용어 정의 검색
- 카테고리별 필터링
- 유사 용어 검색
- JSON 원본 데이터 파싱

사용 예시:
    from tools.glossary_rag import GlossaryRAGTool

    rag = GlossaryRAGTool()
    results = rag.search_term("CTD 문서란?", k=5)

    # JSON 파싱
    term_data = rag.parse_json_content(results[0]['content'])
    print(term_data['description_ko'])
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
        self.task_description = "주어진 의학 및 제약 관련 질문에 대해, 용어집에서 가장 관련성 높은 설명을 검색하시오"

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


class GlossaryRAGTool:
    """
    Qdrant 기반 용어집 검색 도구

    Attributes:
        vector_store: QdrantVectorStore 인스턴스
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        collection_name: str = "glossary_terms",
        model_name: str = "intfloat/multilingual-e5-large-instruct"
    ):
        """
        GlossaryRAGTool 초기화

        Args:
            storage_path: Qdrant 저장 경로 (기본: ./qdrant_storage/glossary)
            collection_name: 검색할 컬렉션 이름
            model_name: 임베딩 모델 이름
        """
        if storage_path is None:
            storage_path = str(project_root / "qdrant_storage" / "glossary")

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

        print(f"✅ GlossaryRAGTool 초기화 완료")
        print(f"   - Storage: {storage_path}")
        print(f"   - Collection: {collection_name}\n")

    def search_term(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        용어 검색 (유사도 기반)

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 최소 유사도 점수 (0~1)

        Returns:
            검색 결과 리스트 [{"term", "content", "category", "score", "metadata"}]
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
                "term": doc.metadata.get("term", "N/A"),
                "term_en": doc.metadata.get("term_en", "N/A"),
                "category": doc.metadata.get("category", "N/A"),
                "content": doc.page_content,
                "score": float(score),
                "synonyms": doc.metadata.get("synonyms", []),
                "metadata": doc.metadata
            })

        print(f"📊 검색 완료: {len(formatted_results)}개 결과 반환\n")
        return formatted_results

    def search_by_category(
        self,
        query: str,
        category: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        카테고리별 필터링 검색

        Args:
            query: 검색 쿼리
            category: 필터링할 카테고리 (예: "R&D", "임상_비임상")
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        print(f"🔍 카테고리 필터링 검색: '{query}' [카테고리: {category}]")

        # Qdrant 필터 생성 (중첩 구조 접근)
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.category",
                    match=MatchValue(value=category)
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
                "term": doc.metadata.get("term", "N/A"),
                "term_en": doc.metadata.get("term_en", "N/A"),
                "category": doc.metadata.get("category", "N/A"),
                "content": doc.page_content,
                "score": float(score),
                "synonyms": doc.metadata.get("synonyms", []),
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
                "term": doc.metadata.get("term", "N/A"),
                "term_en": doc.metadata.get("term_en", "N/A"),
                "category": doc.metadata.get("category", "N/A"),
                "content": doc.page_content,
                "synonyms": doc.metadata.get("synonyms", []),
                "metadata": doc.metadata
            })

        print(f"📊 MMR 검색 완료: {len(formatted_results)}개 결과 반환\n")
        return formatted_results

    def get_term_by_exact_match(self, term: str) -> Optional[Dict[str, Any]]:
        """
        정확한 용어 매칭으로 검색

        Args:
            term: 검색할 용어 (예: "CTD", "ADME")

        Returns:
            검색 결과 (없으면 None)
        """
        print(f"🎯 정확한 용어 검색: '{term}'")

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.term",  # 중첩된 구조 접근
                    match=MatchValue(value=term)
                )
            ]
        )

        results = self.vector_store.similarity_search(
            query=term,
            k=1,
            filter=filter_condition
        )

        if results:
            doc = results[0]
            return {
                "term": doc.metadata.get("term", "N/A"),
                "term_en": doc.metadata.get("term_en", "N/A"),
                "category": doc.metadata.get("category", "N/A"),
                "content": doc.page_content,
                "synonyms": doc.metadata.get("synonyms", []),
                "metadata": doc.metadata
            }

        print(f"❌ 용어를 찾을 수 없습니다: '{term}'\n")
        return None

    @staticmethod
    def parse_json_content(content: str) -> Dict[str, Any]:
        """
        검색 결과의 content(JSON 문자열)를 파싱하여 딕셔너리로 반환

        Args:
            content: JSON 문자열 (검색 결과의 'content' 필드)

        Returns:
            파싱된 용어 데이터 딕셔너리

        Example:
            >>> result = rag.search_term("CTD")[0]
            >>> data = rag.parse_json_content(result['content'])
            >>> print(data['description_ko'])
            '의약품 품목허가 신청 시 자료 구성을 표준화한 국제 공통 양식.'
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 파싱 오류: {e}")
            return {}

    def get_description(self, term: str) -> Optional[str]:
        """
        특정 용어의 설명만 간단히 반환

        Args:
            term: 검색할 용어

        Returns:
            용어 설명 (없으면 None)

        Example:
            >>> rag.get_description("CTD")
            '의약품 품목허가 신청 시 자료 구성을 표준화한 국제 공통 양식.'
        """
        result = self.get_term_by_exact_match(term)
        if result:
            data = self.parse_json_content(result['content'])
            return data.get('description_ko')
        return None


# 사용 예시 및 테스트
if __name__ == "__main__":
    print("=" * 70)
    print("  Glossary RAG Tool 테스트")
    print("=" * 70)
    print()

    # 도구 초기화
    rag = GlossaryRAGTool()

    # 테스트 1: 일반 검색
    print("\n" + "=" * 70)
    print("  테스트 1: 일반 검색")
    print("=" * 70 + "\n")

    results = rag.search_term("CTD 문서란 무엇인가?", k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['term']}] (유사도: {result['score']:.4f})")
        print(f"   카테고리: {result['category']}")
        print(f"   {result['content'][:100]}...")
        print()

    # 테스트 2: 카테고리 필터링
    print("\n" + "=" * 70)
    print("  테스트 2: 카테고리 필터링 검색")
    print("=" * 70 + "\n")

    results = rag.search_by_category("임상시험", category="R&D", k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['term']}] (유사도: {result['score']:.4f})")
        print(f"   {result['content'][:100]}...")
        print()

    # 테스트 3: 정확한 매칭
    print("\n" + "=" * 70)
    print("  테스트 3: 정확한 용어 매칭")
    print("=" * 70 + "\n")

    result = rag.get_term_by_exact_match("ADME")
    if result:
        print(f"✅ 용어 발견: {result['term']} ({result['term_en']})")
        print(f"   JSON 원본: {result['content'][:100]}...")

        # JSON 파싱 테스트
        data = rag.parse_json_content(result['content'])
        print(f"\n   파싱된 데이터:")
        print(f"   - 카테고리: {data.get('category')}")
        print(f"   - 용어: {data.get('term')}")
        print(f"   - 설명: {data.get('description_ko')}")
        print(f"   - 동의어: {data.get('synonyms')}")
    else:
        print("❌ 용어를 찾을 수 없습니다.")

    # 테스트 4: get_description 헬퍼 메서드
    print("\n" + "=" * 70)
    print("  테스트 4: 용어 설명 간단 조회")
    print("=" * 70 + "\n")

    description = rag.get_description("CTD")
    if description:
        print(f"✅ CTD 설명: {description}")
    else:
        print("❌ 용어를 찾을 수 없습니다.")

    print("\n" + "=" * 70)
    print("  테스트 완료")
    print("=" * 70)
