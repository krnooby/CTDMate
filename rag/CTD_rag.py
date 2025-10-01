"""
CTD RAG Tool (Tool2: RegulationTool)

LangGraph StateGraph 기반 규정 검증 도구
- Qdrant Vector 검색 (MFDS/ICH 가이드라인)
- BM25 + MMR 하이브리드 검색
- 체크리스트 자동 검증
- 용어 정규화 (Llama-3.2-1B)
- Self-healing 루프 (검증 → 정규화 → 재검증)

사용 예시:
    from tools.CTD_rag import RegulationTool

    tool = RegulationTool()
    result = tool.run(
        composition_data={"dl_material": "알푸조신염산염", ...},
        ctd_section="M2.7"
    )

    print(result["pass_validation"])
    print(result["violations"])
    print(result["citations"])
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, END
from tools.regulation_state import RegulationState, create_initial_state
from tools.regulation_nodes import (
    validate_node,
    normalize_node,
    expand_coverage_node,
    collect_citations_node
)
from tools.regulation_edges import (
    should_normalize,
    should_expand_coverage,
    should_retry_validation
)


class RegulationTool:
    """
    LangGraph 기반 규정 검증 도구

    Attributes:
        graph: 컴파일된 StateGraph
    """

    def __init__(self):
        """RegulationTool 초기화 (StateGraph 구성)"""
        self.graph = self._build_graph()
        print("✅ RegulationTool (LangGraph) 초기화 완료\n")

    def _build_graph(self) -> Any:
        """
        StateGraph 구성

        Returns:
            컴파일된 그래프
        """
        workflow = StateGraph(RegulationState)

        # ===== 노드 추가 =====
        workflow.add_node("validate", validate_node)
        workflow.add_node("normalize", normalize_node)
        workflow.add_node("expand_coverage", expand_coverage_node)
        workflow.add_node("collect_citations", collect_citations_node)

        # ===== 엔트리 포인트 =====
        workflow.set_entry_point("validate")

        # ===== 조건부 엣지 =====
        # validate → normalize OR expand_coverage
        workflow.add_conditional_edges(
            "validate",
            should_normalize,
            {
                "normalize": "normalize",
                "check_coverage": "expand_coverage"
            }
        )

        # normalize → validate (재검증) OR END
        workflow.add_conditional_edges(
            "normalize",
            should_retry_validation,
            {
                "validate": "validate",
                "__end__": END
            }
        )

        # expand_coverage → expand (재확장) OR collect_citations
        workflow.add_conditional_edges(
            "expand_coverage",
            should_expand_coverage,
            {
                "expand": "expand_coverage",
                "collect_citations": "collect_citations"
            }
        )

        # collect_citations → END
        workflow.add_edge("collect_citations", END)

        # 컴파일
        return workflow.compile()

    def run(
        self,
        composition_data: Dict[str, Any],
        ctd_section: str,
        max_iterations: int = 3
    ) -> RegulationState:
        """
        규정 검증 실행

        Args:
            composition_data: 파싱된 조성 데이터 (SmartDocumentTool 출력)
            ctd_section: 대상 CTD 섹션 (예: "M2.7", "M2.6")
            max_iterations: 최대 반복 횟수 (기본값: 3)

        Returns:
            최종 상태 (RegulationState)

        Example:
            >>> tool = RegulationTool()
            >>> result = tool.run(
            ...     composition_data={"dl_material": "알푸조신염산염"},
            ...     ctd_section="M2.7"
            ... )
            >>> print(result["pass_validation"])
            True
            >>> print(len(result["citations"]))
            5
        """
        print("=" * 70)
        print("  🚀 RegulationTool 실행")
        print("=" * 70)
        print(f"📋 CTD 섹션: {ctd_section}")
        print(f"📊 조성 데이터 필드 수: {len(composition_data)}")
        print(f"🔁 최대 반복 횟수: {max_iterations}\n")

        # 초기 상태 생성
        initial_state = create_initial_state(
            composition_data=composition_data,
            ctd_section=ctd_section,
            max_iterations=max_iterations
        )

        # 그래프 실행
        final_state = self.graph.invoke(initial_state)

        # 결과 요약
        print("\n" + "=" * 70)
        print("  📊 최종 결과")
        print("=" * 70)
        print(f"✅ 검증 통과: {final_state.get('pass_validation', False)}")
        print(f"⚠️  위반 수: {len(final_state.get('violations', []))}")
        print(f"🔧 정규화 적용: {final_state.get('normalization_applied', False)}")
        print(f"📚 검색 스니펫: {len(final_state.get('retrieved_snippets', []))}개")
        print(f"📈 커버리지: {final_state.get('coverage_score', 0.0):.2%}")
        print(f"📖 인용 수: {len(final_state.get('citations', []))}개")
        print(f"🔁 실행 반복: {final_state.get('current_iteration', 0)}회")
        print("=" * 70 + "\n")

        return final_state

    def get_violations(self, state: RegulationState) -> list:
        """위반 사항 리스트 반환"""
        return state.get("violations", [])

    def get_citations(self, state: RegulationState) -> list:
        """인용 메타데이터 리스트 반환"""
        return state.get("citations", [])

    def is_valid(self, state: RegulationState) -> bool:
        """검증 통과 여부"""
        return state.get("pass_validation", False)


# ========== 사용 예시 및 테스트 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("  RegulationTool 테스트")
    print("=" * 70)
    print()

    # 테스트 데이터 (약품 정보)
    test_composition = {
        "dl_name": "자트랄엑스엘정 10mg",
        "dl_name_en": "Xatral XL Tab. 10mg",
        "dl_material": "알푸조신염산염",
        "dl_material_en": "Alfuzosin Hydrochloride",
        "dl_custom_shape": "서방형정제",
        "dl_company": "(주)한독",
        "dl_company_en": "Handok",
        "drug_shape": "원형",
        "color_class1": "하양, 노랑",
        "color_class2": "노랑",
        "thick": 8,
        "leng_long": 8,
        "leng_short": 8
    }

    # Tool 초기화
    tool = RegulationTool()

    # 실행
    result = tool.run(
        composition_data=test_composition,
        ctd_section="2.7",  # 임상 요약
        max_iterations=3
    )

    # 결과 출력
    print("\n📋 상세 결과:")
    print(f"\n1. 검증 통과: {tool.is_valid(result)}")

    violations = tool.get_violations(result)
    if violations:
        print(f"\n2. 위반 사항 ({len(violations)}개):")
        for i, v in enumerate(violations, 1):
            print(f"   {i}. [{v['severity'].upper()}] {v['rule']}")
            print(f"      필드: {v['field']} = {v['current_value']}")

    citations = tool.get_citations(result)
    if citations:
        print(f"\n3. 인용 ({len(citations)}개):")
        for i, c in enumerate(citations[:5], 1):  # 최대 5개만 출력
            print(f"   {i}. {c['source']} - {c['section']}: {c['title']}")

    print("\n" + "=" * 70)
    print("  테스트 완료")
    print("=" * 70)
