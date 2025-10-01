"""
RegulationTool Conditional Edges

LangGraph StateGraph의 조건부 엣지 함수들
상태에 따라 다음 노드를 동적으로 결정
"""

from typing import Literal
from tools.regulation_state import RegulationState


def should_normalize(state: RegulationState) -> Literal["normalize", "check_coverage"]:
    """
    검증 후 정규화 필요 여부 판단

    Args:
        state: 현재 상태

    Returns:
        "normalize": Major 위반 있음, 정규화 필요
        "check_coverage": 위반 없거나 Minor만 있음, 커버리지 확인으로
    """
    # Major 위반이 있으면 정규화 노드로
    if state.get("violation_severity") == "major":
        print("→ [ROUTER] Major 위반 감지 → normalize_node")
        return "normalize"

    # 그 외에는 커버리지 확인으로
    print("→ [ROUTER] 위반 없음 또는 Minor → expand_coverage_node")
    return "check_coverage"


def should_expand_coverage(state: RegulationState) -> Literal["expand", "collect_citations"]:
    """
    커버리지 확장 필요 여부 판단

    Args:
        state: 현재 상태

    Returns:
        "expand": 커버리지 부족, 추가 검색 필요
        "collect_citations": 커버리지 충분, 인용 수집으로
    """
    coverage = state.get("coverage_score", 0.0)
    iteration = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # 커버리지 부족하고 반복 한도 내면 재확장
    if state.get("need_expand_coverage", False) and iteration < max_iter:
        print(f"→ [ROUTER] 커버리지 {coverage:.2%} < 80% (반복 {iteration}/{max_iter}) → expand_coverage_node")
        return "expand"

    # 충분하거나 한도 도달하면 인용 수집으로
    print(f"→ [ROUTER] 커버리지 {coverage:.2%} 충족 또는 한도 도달 → collect_citations_node")
    return "collect_citations"


def should_retry_validation(state: RegulationState) -> Literal["validate", "__end__"]:
    """
    정규화 후 재검증 필요 여부 판단

    Args:
        state: 현재 상태

    Returns:
        "validate": 정규화 적용됨, 재검증 필요
        "__end__": 정규화 미적용 또는 반복 한도, 종료
    """
    iteration = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # 정규화가 적용되었고 반복 한도 내면 재검증
    if state.get("normalization_applied", False) and iteration < max_iter:
        print(f"→ [ROUTER] 정규화 적용됨 (반복 {iteration}/{max_iter}) → validate_node 재실행")
        return "validate"

    # 그 외에는 종료
    print("→ [ROUTER] 정규화 미적용 또는 한도 도달 → 종료")
    return "__end__"
