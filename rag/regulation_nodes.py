"""
RegulationTool Nodes

LangGraph StateGraph의 노드 함수들
- validate_node: 규정 검증 (체크리스트 매칭)
- normalize_node: 용어 정규화 (Llama-3.2-1B)
- expand_coverage_node: 스니펫 확장 (BM25 + Vector + MMR)
- collect_citations_node: 인용 수집
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import re

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.regulation_state import RegulationState, Violation
from tools.mfds_rag import MFDSRAGTool
from tools.glossary_rag import GlossaryRAGTool
from tools.bm25_search import BM25Search


def validate_node(state: RegulationState) -> RegulationState:
    """
    규정 검증 노드

    MFDS RAG로 체크리스트를 검색하고, composition_data와 비교하여 위반 사항 탐지

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태 (violations, violation_severity, pass_validation)
    """
    print(f"\n🔍 [VALIDATE] CTD 섹션 {state['ctd_section']} 검증 시작...")

    # 1. MFDS RAG로 체크리스트 검색
    rag = MFDSRAGTool()
    section_info = rag.get_section_by_id(state["ctd_section"])

    if not section_info:
        print(f"⚠️  섹션 {state['ctd_section']}을 찾을 수 없습니다. 검증 스킵.")
        return {
            **state,
            "validated": True,
            "pass_validation": True,
            "violations": [],
            "violation_severity": "none"
        }

    checklist = section_info.get("checklist", [])
    print(f"📋 체크리스트 항목 수: {len(checklist)}")

    # 2. 각 체크리스트 항목 검증
    violations: List[Violation] = []
    composition = state["composition_data"]

    for item in checklist:
        violation = _check_compliance(composition, item, state["ctd_section"])
        if violation:
            violations.append(violation)

    # 3. 위반 심각도 판단
    severity = _determine_severity(violations)
    print(f"📊 검증 결과: {len(violations)}개 위반 (심각도: {severity})")

    # 4. 상태 업데이트
    return {
        **state,
        "validated": True,
        "violations": violations,
        "violation_severity": severity,
        "pass_validation": len(violations) == 0,
        "need_normalize": severity == "major"
    }


def normalize_node(state: RegulationState) -> RegulationState:
    """
    용어 정규화 노드

    Major 위반 사항의 필드를 표준 형식으로 정규화
    현재는 glossary RAG 기반, 추후 Llama-3.2-1B LoRA로 교체 예정

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태 (normalized_fields, normalization_applied)
    """
    print(f"\n🔧 [NORMALIZE] 용어 정규화 시작 (반복 {state['current_iteration'] + 1}/{state['max_iterations']})...")

    normalized = {}
    glossary_rag = GlossaryRAGTool()

    for violation in state["violations"]:
        if violation["severity"] == "major":
            field = violation["field"]
            current_value = violation["current_value"]

            # Glossary RAG로 표준 용어 검색
            results = glossary_rag.search_term(str(current_value), k=1, score_threshold=0.7)

            if results:
                # 가장 유사한 용어의 영문명 사용
                term_data = glossary_rag.parse_json_content(results[0]['content'])
                normalized[field] = term_data.get('term_en', current_value)
                print(f"  ✓ {field}: '{current_value}' → '{normalized[field]}'")
            else:
                print(f"  ⚠️  {field}: '{current_value}' - 표준 용어 미발견")

    print(f"📊 정규화 완료: {len(normalized)}개 필드")

    return {
        **state,
        "normalized_fields": {**state.get("normalized_fields", {}), **normalized},
        "normalization_applied": len(normalized) > 0,
        "current_iteration": state["current_iteration"] + 1
    }


def expand_coverage_node(state: RegulationState) -> RegulationState:
    """
    커버리지 확장 노드

    BM25 + Vector + MMR 하이브리드 검색으로 스니펫 확장

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태 (retrieved_snippets, coverage_score)
    """
    print(f"\n📚 [EXPAND] 커버리지 확장 중 (BM25 + Vector + MMR)...")

    query = f"{state['ctd_section']} 작성 요구사항"

    # 1. BM25 키워드 검색
    bm25 = BM25Search()
    bm25_results = bm25.search(query, k=10)
    print(f"  ✓ BM25 검색: {len(bm25_results)}개")

    # 2. Vector 검색 (MFDS 가이드라인)
    rag = MFDSRAGTool()
    vector_results = rag.search_guideline(query, k=10)
    print(f"  ✓ Vector 검색: {len(vector_results)}개")

    # 3. 결과 병합 (중복 제거, 점수 기반)
    combined = _merge_search_results(bm25_results, vector_results)
    print(f"  ✓ 병합 결과: {len(combined)}개")

    # 4. MMR 다양성 필터링
    mmr_results = rag.search_with_mmr(
        query=query,
        k=5,
        fetch_k=20,
        lambda_mult=0.5  # 관련성 50%, 다양성 50%
    )
    print(f"  ✓ MMR 필터링: {len(mmr_results)}개")

    # 5. 기존 스니펫과 병합 (중복 제거)
    existing_ids = {s.get("section") for s in state.get("retrieved_snippets", [])}
    new_snippets = [s for s in mmr_results if s.get("section") not in existing_ids]

    # BM25 결과 중 상위 점수도 추가
    for r in combined[:3]:
        if r.get("section") not in existing_ids:
            new_snippets.append(r)
            existing_ids.add(r.get("section"))

    all_snippets = state.get("retrieved_snippets", []) + new_snippets

    # 6. 커버리지 점수 계산
    coverage = _calculate_coverage(state["composition_data"], all_snippets)

    print(f"📊 스니펫 추가: {len(new_snippets)}개 (총 {len(all_snippets)}개)")
    print(f"📈 커버리지: {coverage:.2%}")

    return {
        **state,
        "retrieved_snippets": all_snippets,
        "coverage_score": coverage,
        "need_expand_coverage": coverage < 0.8 and state["current_iteration"] < state["max_iterations"]
    }


def collect_citations_node(state: RegulationState) -> RegulationState:
    """
    인용 수집 노드

    검색된 스니펫에서 인용 메타데이터 추출

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태 (citations)
    """
    print(f"\n📖 [CITATIONS] 인용 정보 수집 중...")

    citations = []
    for snippet in state.get("retrieved_snippets", []):
        citations.append({
            "source": f"MFDS {snippet.get('module', 'N/A')}",
            "module": snippet.get("module", "N/A"),
            "section": snippet.get("section", "N/A"),
            "title": snippet.get("title", "N/A"),
            "snippet_id": snippet.get("metadata", {}).get("para_id", "N/A"),
            "page": None  # MFDS 데이터에는 페이지 정보 없음
        })

    print(f"📊 인용 수집 완료: {len(citations)}개")

    return {
        **state,
        "citations": citations
    }


# ========== Helper Functions ==========

def _check_compliance(composition: Dict[str, Any], rule: str, section: str) -> Violation | None:
    """
    단일 체크리스트 규칙 검증

    Args:
        composition: 조성 데이터
        rule: 체크리스트 규칙 문자열
        section: CTD 섹션

    Returns:
        위반 사항 또는 None
    """
    # 규칙에서 필수 필드 추출 (간단한 키워드 매칭)
    required_fields = _extract_required_fields(rule)

    for field in required_fields:
        # composition에 필드가 없거나 빈 값이면 위반
        value = composition.get(field)
        if not value or (isinstance(value, str) and not value.strip()):
            return Violation(
                rule=rule,
                severity="major" if "must" in rule.lower() or "필수" in rule else "minor",
                field=field,
                current_value=value,
                expected_format=rule
            )

    return None


def _extract_required_fields(rule: str) -> List[str]:
    """
    체크리스트 규칙에서 필수 필드명 추출

    예: "성분명 및 함량 명시" → ["dl_material", "dl_material_en"]

    Args:
        rule: 체크리스트 규칙 문자열

    Returns:
        필드명 리스트
    """
    # 간단한 키워드 매핑 (실제로는 더 정교한 매핑 필요)
    keyword_mapping = {
        "성분": ["dl_material", "dl_material_en"],
        "제형": ["dl_custom_shape"],
        "제조": ["dl_company", "dl_company_en"],
        "용량": ["leng_long", "leng_short", "thick"],
        "색상": ["color_class1", "color_class2"],
    }

    fields = []
    for keyword, field_names in keyword_mapping.items():
        if keyword in rule:
            fields.extend(field_names)

    return fields if fields else ["dl_material"]  # 기본값


def _determine_severity(violations: List[Violation]) -> str:
    """
    위반 리스트에서 전체 심각도 판단

    Args:
        violations: 위반 사항 리스트

    Returns:
        "major" | "minor" | "none"
    """
    if not violations:
        return "none"

    has_major = any(v["severity"] == "major" for v in violations)
    return "major" if has_major else "minor"


def _merge_search_results(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6
) -> List[Dict[str, Any]]:
    """
    BM25와 Vector 검색 결과 병합 (가중 점수 기반)

    Args:
        bm25_results: BM25 검색 결과
        vector_results: Vector 검색 결과
        bm25_weight: BM25 점수 가중치 (기본: 0.4)
        vector_weight: Vector 점수 가중치 (기본: 0.6)

    Returns:
        병합된 결과 리스트 (점수 기준 내림차순)
    """
    # 섹션별로 점수 집계
    score_map = {}

    # BM25 점수 추가
    for r in bm25_results:
        section = r.get("section")
        if section:
            score_map[section] = {
                "data": r,
                "bm25_score": r.get("score", 0.0) * bm25_weight,
                "vector_score": 0.0
            }

    # Vector 점수 추가 (기존 섹션이면 vector_score만 업데이트)
    for r in vector_results:
        section = r.get("section")
        if section:
            if section in score_map:
                score_map[section]["vector_score"] = r.get("score", 0.0) * vector_weight
            else:
                score_map[section] = {
                    "data": r,
                    "bm25_score": 0.0,
                    "vector_score": r.get("score", 0.0) * vector_weight
                }

    # 합산 점수 계산 및 정렬
    merged = []
    for section, info in score_map.items():
        combined_score = info["bm25_score"] + info["vector_score"]
        result = info["data"].copy()
        result["combined_score"] = combined_score
        merged.append(result)

    # 점수 내림차순 정렬
    merged.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)

    return merged


def _calculate_coverage(composition: Dict[str, Any], snippets: List[Dict[str, Any]]) -> float:
    """
    스니펫이 composition 데이터를 얼마나 커버하는지 계산

    간단한 휴리스틱: 스니펫 개수 기반 (실제로는 더 정교한 로직 필요)

    Args:
        composition: 조성 데이터
        snippets: 검색된 스니펫 리스트

    Returns:
        커버리지 점수 (0~1)
    """
    if not snippets:
        return 0.0

    # 간단 휴리스틱: 스니펫 5개 이상이면 80% 커버
    return min(len(snippets) / 5.0, 1.0) * 0.8
