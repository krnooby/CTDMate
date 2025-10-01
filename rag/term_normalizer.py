"""
Term Normalizer (용어 정규화)

Llama-3.2-1B-Instruct 기반 의약품 용어 정규화 모듈
- 한글 성분명 → 영문 표준명 변환
- Glossary RAG 기반 in-context learning (현재)
- 추후 LoRA 파인튜닝 모델로 교체 예정

사용 예시:
    from tools.term_normalizer import TermNormalizer

    normalizer = TermNormalizer()
    result = normalizer.normalize("알푸조신염산염")
    # → "Alfuzosin Hydrochloride"
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.glossary_rag import GlossaryRAGTool


class TermNormalizer:
    """
    용어 정규화 모듈

    현재: Glossary RAG 기반 검색
    향후: Llama-3.2-1B LoRA 파인튜닝 모델 통합

    Attributes:
        glossary_rag: GlossaryRAGTool 인스턴스
        llm: Llama-3.2-1B 모델 (추후 통합)
    """

    def __init__(self, use_llm: bool = False):
        """
        TermNormalizer 초기화

        Args:
            use_llm: Llama-3.2-1B 사용 여부 (현재는 False, 추후 구현)
        """
        self.use_llm = use_llm
        self.glossary_rag = GlossaryRAGTool()

        # Llama-3.2-1B 모델 (추후 통합)
        self.llm = None
        if use_llm:
            print("⚠️  Llama-3.2-1B 모델은 아직 구현되지 않았습니다. Glossary RAG를 사용합니다.")
            # TODO: llama-cpp-python으로 모델 로드
            # from llama_cpp import Llama
            # self.llm = Llama(model_path="models/llama-3.2-1b-term-normalizer.gguf")

        print("✅ TermNormalizer 초기화 완료 (모드: {})".format(
            "LLM" if use_llm and self.llm else "Glossary RAG"
        ))

    def normalize(
        self,
        term: str,
        context: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.7
    ) -> str:
        """
        용어 정규화 (한글 → 영문 표준명)

        Args:
            term: 정규화할 용어 (예: "알푸조신염산염", "락토오스일수화물")
            context: 추가 컨텍스트 정보 (선택, 추후 LLM 프롬프트에 활용)
            score_threshold: RAG 검색 최소 유사도 (기본: 0.7)

        Returns:
            정규화된 영문 표준명 (실패 시 원본 반환)

        Example:
            >>> normalizer = TermNormalizer()
            >>> normalizer.normalize("알푸조신염산염")
            'Alfuzosin Hydrochloride'
        """
        # LLM 모드 (추후 구현)
        if self.use_llm and self.llm:
            return self._normalize_with_llm(term, context)

        # Glossary RAG 모드 (현재)
        return self._normalize_with_rag(term, score_threshold)

    def normalize_batch(
        self,
        terms: list[str],
        score_threshold: float = 0.7
    ) -> Dict[str, str]:
        """
        배치 정규화

        Args:
            terms: 정규화할 용어 리스트
            score_threshold: RAG 검색 최소 유사도

        Returns:
            {원본: 정규화} 매핑 딕셔너리

        Example:
            >>> normalizer = TermNormalizer()
            >>> result = normalizer.normalize_batch(["알푸조신염산염", "락토오스"])
            >>> print(result)
            {'알푸조신염산염': 'Alfuzosin Hydrochloride', '락토오스': 'Lactose'}
        """
        results = {}
        for term in terms:
            results[term] = self.normalize(term, score_threshold=score_threshold)

        return results

    def _normalize_with_rag(self, term: str, score_threshold: float) -> str:
        """
        Glossary RAG 기반 정규화

        Args:
            term: 정규화할 용어
            score_threshold: 최소 유사도

        Returns:
            영문 표준명 (실패 시 원본)
        """
        # 1. Glossary RAG 검색
        results = self.glossary_rag.search_term(
            query=term,
            k=1,
            score_threshold=score_threshold
        )

        if not results:
            print(f"⚠️  '{term}': 표준 용어 미발견 (임계값: {score_threshold})")
            return term

        # 2. 가장 유사한 용어의 영문명 추출
        top_result = results[0]
        term_data = self.glossary_rag.parse_json_content(top_result['content'])

        normalized = term_data.get('term_en')
        if not normalized:
            print(f"⚠️  '{term}': 영문명 필드 없음")
            return term

        print(f"✓ '{term}' → '{normalized}' (유사도: {top_result['score']:.3f})")
        return normalized

    def _normalize_with_llm(
        self,
        term: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Llama-3.2-1B 기반 정규화 (추후 구현)

        Args:
            term: 정규화할 용어
            context: 추가 컨텍스트

        Returns:
            영문 표준명
        """
        # TODO: Llama-3.2-1B LoRA 모델 추론
        # prompt = f"""다음 의약품 성분명을 표준 영문명으로 변환하시오.
        #
        # 입력: {term}
        # 출력:"""
        #
        # response = self.llm(prompt, max_tokens=50, stop=["\n"])
        # return response["choices"][0]["text"].strip()

        # 임시: RAG 폴백
        return self._normalize_with_rag(term, score_threshold=0.7)


# ========== 사용 예시 및 테스트 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("  TermNormalizer 테스트")
    print("=" * 70)
    print()

    normalizer = TermNormalizer()

    # 테스트 1: 단일 정규화
    print("\n[테스트 1] 단일 용어 정규화")
    print("-" * 70)
    test_terms = [
        "알푸조신염산염",
        "락토오스일수화물",
        "마그네슘스테아레이트",
        "전분",
        "존재하지않는용어"
    ]

    for term in test_terms:
        result = normalizer.normalize(term)
        print(f"  • {term} → {result}")

    # 테스트 2: 배치 정규화
    print("\n[테스트 2] 배치 정규화")
    print("-" * 70)
    batch_terms = ["알푸조신염산염", "락토오스", "전분"]
    batch_results = normalizer.normalize_batch(batch_terms)

    print("\n배치 결과:")
    for original, normalized in batch_results.items():
        print(f"  • {original} → {normalized}")

    # 테스트 3: 낮은 임계값
    print("\n[테스트 3] 낮은 임계값 테스트 (0.5)")
    print("-" * 70)
    result_low = normalizer.normalize("락토스", score_threshold=0.5)
    print(f"  • 락토스 → {result_low}")

    print("\n" + "=" * 70)
    print("  테스트 완료")
    print("=" * 70)
