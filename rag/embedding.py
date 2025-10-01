# 파일명: embedding_util.py

from sentence_transformers import SentenceTransformer
import torch
from typing import List

class E5Embedder:
    """
    SentenceTransformer를 사용하여 intfloat/multilingual-e5-large-instruct 모델을 관리하고,
    문서와 "Instruct/Query" 형식의 질문에 대한 임베딩을 생성하는 클래스.
    """
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large-instruct', device: str = None):
        """
        모델을 초기화하고 지정된 디바이스에 로드합니다.
        모델 로드는 이 클래스의 인스턴스를 생성할 때 한 번만 수행됩니다.
        
        :param model_name: Hugging Face에 등록된 모델 이름
        :param device: 모델을 로드할 디바이스 ('cuda', 'cpu' 등). None이면 자동 선택됩니다.
        """
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ '{model_name}' 모델이 [{self.model.device}] 디바이스에 성공적으로 로드되었습니다.")

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        """E5-instruct 모델을 위한 고유의 프롬프트 형식을 생성합니다."""
        return f'Instruct: {task_description}\nQuery: {query}'

    def embed_documents(self, documents: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        """
        주어진 문서 리스트를 임베딩합니다. 
        문서 임베딩 시에는 별도의 instruction 프롬프트가 필요 없습니다.

        :param documents: 임베딩할 텍스트(문서) 리스트
        :param batch_size: 한 번에 처리할 문서 수
        :param show_progress_bar: 임베딩 진행 상태를 터미널에 표시할지 여부
        :return: 문서 임베딩 텐서
        """
        print(f"총 {len(documents)}개의 문서를 E5 모델로 임베딩합니다...")
        return self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            normalize_embeddings=True  # E5 모델은 정규화가 권장됩니다.
        )

    def embed_queries(self, queries: List[str], task: str, batch_size: int = 32) -> torch.Tensor:
        """
        질문(Query) 리스트를 "Instruct/Query" 형식으로 변환한 후 임베딩합니다.
        
        :param queries: 임베딩할 사용자 질문 리스트
        :param task: 질문의 의도를 설명하는 한 문장의 instruction
        :param batch_size: 한 번에 처리할 질문 수
        :return: 질문 임베딩 텐서
        """
        # Instruction을 각 query에 적용하여 모델에 입력할 새로운 리스트를 생성합니다.
        instruct_queries = [self._get_detailed_instruct(task, q) for q in queries]
        
        print(f"총 {len(queries)}개의 질문을 E5 모델로 임베딩합니다...")
        return self.model.encode(
            instruct_queries,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

# 이 파일이 직접 실행될 경우 아래의 테스트 코드가 동작합니다.
if __name__ == '__main__':
    # 클래스 인스턴스 생성
    embedder = E5Embedder()

    # --- 테스트용 데이터 (영어, 한글 제약/의학 용어 예시로 변경) ---
    test_docs = [
        # 한글 문서 예시
        "카테고리: 임상_비임상\n용어: 임상시험 심사위원회 (Institutional Review Board)\n설명: 피험자 보호와 연구윤리를 심의·승인하는 독립 위원회.\n동의어: IRB",
        # 영어 문서 예시
        "Category: R&D\nTerm: ADME (Absorption, Distribution, Metabolism, Excretion)\nDescription: A core process in pharmacology and drug development that describes the disposition of a pharmaceutical compound within an organism.\nSynonyms: Pharmacokinetics"
    ]
    
    test_queries = [
        # 한글 질문 예시 (첫 번째 문서와 관련)
        "피험자 보호를 위해 연구윤리를 심의하는 위원회는 무엇인가요?",
        # 영어 질문 예시 (두 번째 문서와 관련)
        "What does ADME stand for in drug development?"
    ]
    
    # 모델에 전달할 작업 설명 (Task Description)
    retrieval_task = '주어진 의학 및 제약 관련 질문에 대해, 용어집에서 가장 관련성 높은 설명을 검색하시오'

    # 임베딩 실행
    doc_embeddings = embedder.embed_documents(test_docs)
    query_embeddings = embedder.embed_queries(test_queries, task=retrieval_task)

    print("\n--- 모듈 테스트 결과 ---")
    print("문서 임베딩 Shape:", doc_embeddings.shape)
    print("질문 임베딩 Shape:", query_embeddings.shape)
    
    # 유사도 계산 (각 질문과 모든 문서 간의 유사도)
    # 결과 행렬: 각 행은 하나의 질문, 각 열은 해당 질문과 문서 간의 유사도 점수
    scores = (query_embeddings @ doc_embeddings.T)
    print("\n유사도 점수 (각 질문 vs 모든 문서):")
    print(scores.tolist())
