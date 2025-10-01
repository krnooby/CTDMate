"""
BM25 Search Module

MFDS/ICH 가이드라인 문서에 대한 BM25 키워드 검색
- rank_bm25 라이브러리 사용
- 한글 형태소 분석 (간단한 공백 기반 토크나이징)
- JSONL 데이터 인덱싱
- Vector 검색과 하이브리드 사용

사용 예시:
    from tools.bm25_search import BM25Search

    bm25 = BM25Search()
    results = bm25.search("임상 요약 작성", k=5)
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rank_bm25 import BM25Okapi


class BM25Search:
    """
    BM25 기반 키워드 검색 도구

    Attributes:
        corpus: 문서 리스트
        bm25: BM25Okapi 인스턴스
        documents: 원본 문서 메타데이터
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        index_path: Optional[str] = None
    ):
        """
        BM25Search 초기화

        Args:
            data_path: MFDS JSONL 데이터 경로 (기본: data/MFDS/MFDS_final.jsonl)
            index_path: BM25 인덱스 저장 경로 (기본: qdrant_storage/mfds_bm25.pkl)
        """
        if data_path is None:
            data_path = str(project_root / "data" / "MFDS" / "MFDS_final.jsonl")

        if index_path is None:
            index_path = str(project_root / "qdrant_storage" / "mfds_bm25.pkl")

        self.data_path = data_path
        self.index_path = index_path
        self.corpus: List[List[str]] = []
        self.documents: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

        # 인덱스 로드 또는 구축
        if Path(index_path).exists():
            print(f"📂 BM25 인덱스 로드 중... ({index_path})")
            self._load_index()
        else:
            print(f"🔨 BM25 인덱스 구축 중... ({data_path})")
            self._build_index()
            self._save_index()

        print(f"✅ BM25Search 초기화 완료 (문서 수: {len(self.documents)})\n")

    def _build_index(self):
        """MFDS JSONL 데이터로부터 BM25 인덱스 구축"""
        # JSONL 읽기
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 빈 줄 건너뛰기
                    continue
                doc = json.loads(line)
                self.documents.append(doc)

                # 검색 대상 텍스트 생성 (제목 + 설명 + 체크리스트)
                search_text = self._create_search_text(doc)
                tokens = self._tokenize(search_text)
                self.corpus.append(tokens)

        # BM25 인덱스 구축
        self.bm25 = BM25Okapi(self.corpus)
        print(f"📊 BM25 인덱스 구축 완료: {len(self.documents)}개 문서")

    def _create_search_text(self, doc: Dict[str, Any]) -> str:
        """
        문서에서 검색 대상 텍스트 생성

        Args:
            doc: MFDS 문서 딕셔너리

        Returns:
            통합 검색 텍스트
        """
        parts = [
            doc.get("title", ""),
            doc.get("description", ""),
            " ".join(doc.get("checklist", []))
        ]
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """
        간단한 토크나이징 (공백 기반)

        추후 KoNLPy 등으로 개선 가능

        Args:
            text: 입력 텍스트

        Returns:
            토큰 리스트
        """
        # 간단한 공백 분리 + 특수문자 제거
        tokens = text.lower().split()
        return [t.strip(".,!?()[]{}:;") for t in tokens if len(t) > 1]

    def _save_index(self):
        """BM25 인덱스 저장 (pickle)"""
        index_dir = Path(self.index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'documents': self.documents,
                'bm25': self.bm25
            }, f)

        print(f"💾 BM25 인덱스 저장: {self.index_path}")

    def _load_index(self):
        """BM25 인덱스 로드 (pickle)"""
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            self.corpus = data['corpus']
            self.documents = data['documents']
            self.bm25 = data['bm25']

        print(f"✅ BM25 인덱스 로드 완료: {len(self.documents)}개 문서")

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        BM25 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 최소 BM25 점수 (기본: 0.0, 필터링 안함)

        Returns:
            검색 결과 리스트 [{"module", "section", "title", "score", ...}]

        Example:
            >>> bm25 = BM25Search()
            >>> results = bm25.search("임상 요약", k=3)
            >>> for r in results:
            ...     print(f"{r['section']}: {r['title']} (점수: {r['score']:.2f})")
        """
        print(f"🔍 BM25 검색: '{query}' (top-{k})")

        # 쿼리 토크나이징
        query_tokens = self._tokenize(query)

        # BM25 점수 계산
        scores = self.bm25.get_scores(query_tokens)

        # 상위 k개 인덱스 및 점수 추출
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        # 결과 포맷팅
        results = []
        for idx in top_indices:
            score = float(scores[idx])

            # 점수 필터링
            if score < score_threshold:
                continue

            doc = self.documents[idx]
            results.append({
                "module": doc.get("module", "N/A"),
                "section": doc.get("section", "N/A"),
                "title": doc.get("title", "N/A"),
                "description": doc.get("description", "N/A"),
                "checklist": doc.get("checklist", []),
                "cross_ref": doc.get("cross_ref", []),
                "score": score,
                "metadata": doc
            })

        print(f"📊 BM25 검색 완료: {len(results)}개 결과\n")
        return results

    def rebuild_index(self):
        """인덱스 재구축"""
        print("🔄 BM25 인덱스 재구축 중...")
        self.corpus = []
        self.documents = []
        self._build_index()
        self._save_index()


# ========== 사용 예시 및 테스트 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("  BM25Search 테스트")
    print("=" * 70)
    print()

    # BM25Search 초기화
    bm25 = BM25Search()

    # 테스트 1: 기본 검색
    print("\n[테스트 1] 기본 BM25 검색")
    print("-" * 70)
    results = bm25.search("임상 요약 작성 방법", k=5)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['section']}] {r['title']}")
        print(f"   BM25 점수: {r['score']:.4f}")
        print(f"   설명: {r['description'][:60]}...")
        print()

    # 테스트 2: 특정 키워드 검색
    print("\n[테스트 2] 특정 키워드 검색")
    print("-" * 70)
    results2 = bm25.search("약동학 PK ADME", k=3)

    for i, r in enumerate(results2, 1):
        print(f"{i}. [{r['section']}] {r['title']} (점수: {r['score']:.2f})")

    # 테스트 3: 점수 임계값 적용
    print("\n[테스트 3] 점수 임계값 필터링 (threshold=5.0)")
    print("-" * 70)
    results3 = bm25.search("임상시험", k=10, score_threshold=5.0)

    print(f"임계값 이상 결과: {len(results3)}개")
    for i, r in enumerate(results3[:5], 1):
        print(f"{i}. [{r['section']}] {r['title']} (점수: {r['score']:.2f})")

    print("\n" + "=" * 70)
    print("  테스트 완료")
    print("=" * 70)
