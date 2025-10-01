"""
RegulationTool State Schema

LangGraph StateGraph에서 사용하는 규정 검증 상태 정의
Blackboard 패턴으로 노드 간 데이터 공유
"""

from typing import TypedDict, List, Dict, Any, Literal


class RegulationState(TypedDict, total=False):
    """
    RegulationTool의 상태 스키마

    Attributes:
        # 입력 데이터
        composition_data: 파싱된 조성 데이터 (SmartDocumentTool 출력)
        ctd_section: 대상 CTD 섹션 (예: "M2.3", "M2.6", "M2.7")

        # 검증 결과
        validated: 검증 완료 여부
        pass_validation: 검증 통과 여부 (모든 체크리스트 충족)
        violations: 위반 사항 리스트
        violation_severity: 위반 심각도 ("major" | "minor" | "none")

        # RAG 검색 결과
        retrieved_snippets: MFDS/ICH 가이드라인 스니펫
        coverage_score: 검색된 스니펫이 composition_data를 커버하는 비율 (0~1)
        rag_confidence: RAG 검색 신뢰도

        # 정규화 (Normalization)
        normalized_fields: 정규화된 필드 매핑 (예: {"dl_material": "Lactose monohydrate"})
        normalization_applied: 정규화 적용 여부

        # 인용 (Citations)
        citations: 인용 메타데이터 리스트

        # 제어 플래그
        max_iterations: 최대 반복 횟수 (무한 루프 방지)
        current_iteration: 현재 반복 횟수
        need_expand_coverage: 커버리지 확장 필요 여부
        need_normalize: 정규화 필요 여부
    """
    # 입력
    composition_data: Dict[str, Any]
    ctd_section: str

    # 검증
    validated: bool
    pass_validation: bool
    violations: List[Dict[str, Any]]
    violation_severity: Literal["major", "minor", "none"]

    # RAG
    retrieved_snippets: List[Dict[str, Any]]
    coverage_score: float
    rag_confidence: float

    # 정규화
    normalized_fields: Dict[str, str]
    normalization_applied: bool

    # 인용
    citations: List[Dict[str, Any]]

    # 제어
    max_iterations: int
    current_iteration: int
    need_expand_coverage: bool
    need_normalize: bool


class Violation(TypedDict):
    """위반 사항 구조"""
    rule: str                    # 체크리스트 규칙 (예: "Must include manufacturing process")
    severity: Literal["major", "minor"]
    field: str                   # 위반 필드명 (예: "dl_material")
    current_value: Any           # 현재 값
    expected_format: str         # 기대 형식 설명


class Citation(TypedDict):
    """인용 메타데이터 구조"""
    source: str                  # 출처 (예: "ICH M4Q", "MFDS M2.7")
    module: str                  # CTD 모듈 (예: "M2.7")
    section: str                 # 섹션 번호 (예: "2.7.1.2")
    title: str                   # 섹션 제목
    snippet_id: str              # 스니펫 고유 ID
    page: int | None             # 페이지 번호 (있을 경우)


def create_initial_state(
    composition_data: Dict[str, Any],
    ctd_section: str,
    max_iterations: int = 3
) -> RegulationState:
    """
    초기 상태 생성 헬퍼 함수

    Args:
        composition_data: 파싱된 조성 데이터
        ctd_section: 대상 CTD 섹션
        max_iterations: 최대 반복 횟수 (기본값: 3)

    Returns:
        초기화된 RegulationState
    """
    return RegulationState(
        composition_data=composition_data,
        ctd_section=ctd_section,
        validated=False,
        pass_validation=False,
        violations=[],
        violation_severity="none",
        retrieved_snippets=[],
        coverage_score=0.0,
        rag_confidence=0.0,
        normalized_fields={},
        normalization_applied=False,
        citations=[],
        max_iterations=max_iterations,
        current_iteration=0,
        need_expand_coverage=True,
        need_normalize=False
    )
