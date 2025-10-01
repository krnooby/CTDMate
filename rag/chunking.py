"""
용어집 청킹 유틸리티

JSONL 형식의 용어집 파일을 다양한 형태로 청킹합니다.
- JSON 원본 유지
- 구조화된 텍스트 변환
- 딕셔너리 객체 반환
"""

import json
from typing import List, Dict, Any


def chunk_glossary_as_json_string(file_path: str) -> List[str]:
    """
    용어집 JSONL 파일을 읽어 각 JSON 객체를 문자열로 반환

    :param file_path: 처리할 JSONL 파일 경로
    :return: JSON 문자열 리스트 (각 항목이 하나의 청크)

    Example:
        >>> chunks = chunk_glossary_as_json_string('glossary.jsonl')
        >>> chunks[0]
        '{"category": "R&D", "term": "ADME", ...}'
    """
    print(f"📖 '{file_path}' 파일에서 용어집 데이터를 로딩합니다 (JSON 원본 유지)...")
    json_chunks = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    # JSON 유효성 검사
                    json.loads(line)
                    # 원본 JSON 문자열 그대로 저장
                    json_chunks.append(line.strip())

                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON 파싱 오류 (라인 {line_num}): {e}")
                    continue

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: '{file_path}'")
        return []

    print(f"✅ 총 {len(json_chunks)}개의 용어를 로드했습니다.\n")
    return json_chunks


def chunk_glossary_as_dict(file_path: str) -> List[Dict[str, Any]]:
    """
    용어집 JSONL 파일을 읽어 딕셔너리 객체 리스트로 반환

    :param file_path: 처리할 JSONL 파일 경로
    :return: 딕셔너리 리스트

    Example:
        >>> chunks = chunk_glossary_as_dict('glossary.jsonl')
        >>> chunks[0]['term']
        'ADME'
    """
    print(f"📖 '{file_path}' 파일에서 용어집 데이터를 로딩합니다 (딕셔너리)...")
    dict_chunks = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    dict_chunks.append(item)

                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON 파싱 오류 (라인 {line_num}): {e}")
                    continue

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: '{file_path}'")
        return []

    print(f"✅ 총 {len(dict_chunks)}개의 용어를 로드했습니다.\n")
    return dict_chunks


def chunk_glossary_as_text(file_path: str) -> List[str]:
    """
    용어집 JSONL 파일을 읽어 구조화된 텍스트 청크 리스트로 변환
    (기존 방식 - 하위 호환성 유지)

    :param file_path: 처리할 JSONL 파일 경로
    :return: 텍스트 청크(문서) 리스트

    Example:
        >>> chunks = chunk_glossary_as_text('glossary.jsonl')
        >>> print(chunks[0])
        카테고리: R&D
        용어: ADME (Absorption, Distribution, Metabolism, Excretion)
        설명: 흡수·분포·대사·배설...
        동의어: 약동학개요
    """
    print(f"📖 '{file_path}' 파일에서 용어집 데이터를 로딩합니다 (구조화 텍스트)...")
    text_chunks = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    description = item.get('description_ko') or item.get('definition_ko', '설명 정보 없음')
                    synonyms_list = item.get('synonyms') or []
                    synonyms_str = ', '.join(synonyms_list) if synonyms_list else '없음'

                    chunk_text = (
                        f"카테고리: {item.get('category', '정보 없음')}\n"
                        f"용어: {item.get('term', '정보 없음')} ({item.get('term_en', '정보 없음')})\n"
                        f"설명: {description}\n"
                        f"동의어: {synonyms_str}"
                    )
                    text_chunks.append(chunk_text)

                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON 파싱 오류 (라인 {line_num}): {e}")
                    continue

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: '{file_path}'")
        return []

    print(f"✅ 총 {len(text_chunks)}개의 용어를 로드했습니다.\n")
    return text_chunks


# 하위 호환성을 위한 별칭
chunk_glossary_jsonl = chunk_glossary_as_text


if __name__ == "__main__":
    """모듈 테스트"""
    test_file = 'data/glossary/glossary_final.jsonl'

    print("=" * 70)
    print("  Chunking Utils 테스트")
    print("=" * 70)
    print()

    # 테스트 1: JSON 문자열
    print("테스트 1: JSON 문자열 청킹")
    print("-" * 70)
    json_chunks = chunk_glossary_as_json_string(test_file)
    if json_chunks:
        print(f"첫 번째 청크 (JSON):\n{json_chunks[0]}\n")

    # 테스트 2: 딕셔너리
    print("\n테스트 2: 딕셔너리 청킹")
    print("-" * 70)
    dict_chunks = chunk_glossary_as_dict(test_file)
    if dict_chunks:
        print(f"첫 번째 청크 (Dict):\n{dict_chunks[0]}\n")

    # 테스트 3: 텍스트 (기존 방식)
    print("\n테스트 3: 텍스트 청킹 (기존 방식)")
    print("-" * 70)
    text_chunks = chunk_glossary_as_text(test_file)
    if text_chunks:
        print(f"첫 번째 청크 (Text):\n{text_chunks[0]}\n")

    print("=" * 70)
    print("테스트 완료")
    print("=" * 70)