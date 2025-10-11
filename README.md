# CTDMate

이 프로젝트는 **CTD(Common Technical Document)** 인허가 문서의 **자동 작성, 매핑, 검증**을 수행하는  
**규칙(Rule) + RAG(검색증강) + LLM 기반 에이전트 시스템**입니다.  

LLM은 선택적으로 사용되며, **온디바이스 환경(Llama-3.2-1B-Instruct 모델 + Llama.cpp)** 에서도 동작 가능합니다.

---

## 🚀 주요 기능

- **Tool 1: 문서 자동 파싱**
  - Upstage Document AI를 사용하여 인허가 문서를 구조화 (JSON, 텍스트 등)
- **Tool 2: CTD 내용 검증**
  - RAG 기반으로 **ICH / MFDS Reference**와 비교하여 누락된 근거 탐지
- **Tool 3: CTD 문서 작성**
  - LLM(Solar Pro 2 / Llama-3.2-1B-Instruct 모델 + Llama.cpp)을 활용한 CTD 문서 자동 생성 및 수정 제안
- **YAML 기반 규칙 점검**
  - 사전 정의된 템플릿 규칙에 따라 자동 검증 수행
- **선택적 LLM 사용**
  - 초안 작성, 용어 통일, 문장 자연화 등 보조 기능 수행

---

## 🧩 시스템 구성 (온디바이스 + 보조 분기)

```
                 ┌─────────────────────────┐
                 │       Input File        │
                 │   (Module1, 2, 3 등)    │
                 └─────────────┬───────────┘
                               │
                               ▼
                      ┌───────────────┐
                      │  LLM Router   │
                      │  (온디바이스) │
                      └───────┬───────┘
                              │
      ┌───────────────────────┼──────────────────────────────────┐
      ▼                       ▼                                  ▼
┌───────────────┐      ┌─────────────────────────┐   ┌─────────────────────────┐
│    Tool 1     │      │         Tool 2          │   │          Tool 3         │
│   문서 파싱   │      │      CTD 내용 검증      │    │      CTD 작성/요약      │
│ Upstage DocAI │      │     RAG Validator       │   │        LLM Writer       │
│   (클라우드)  │      │  (ICH / MFDS Reference) │   │    (클라우드/온디바이스) │
└───────┬───────┘      └─────────┬───────────────┘   └─────────┬───────────────┘
        │                        │                             │
        └────────────────────────┴─┬───────────────────────────┘
                                   ▼                         
                 ┌───────────────────────────────┐
                 │      YAML Rule Checker        │
                 │      + LLM Review Loop        │
                 └───────────────┬───────────────┘
                                 ▼
                   ┌──────────────────────────────┐
                   │부족한 근거/누락 시 루프 재진입│
                   │ (RAG / LLM / 규칙 단계 복귀) │
                   └───────────────┬──────────────┘
                                   ▼
                            ┌───────────┐
                            │   Output  │
                            │(CTD 파일) │
                            └───────────┘
```


## 🧠 LLM 활용 전략

| 환경        | 모델                                         | 역할 |
|------------|--------------------------------------------|------|
| 온디바이스 | **Llama-3.2-1B-Instruct + Llama.cpp**      | 로컬 환경에서 LLM 기반 문서 생성 및 보정 |
| 클라우드   | **Solar Pro 2 (Upstage)**                  | 초안 작성, 요약, 용어 통일 |

---

## ⚙️ 구성 모듈 상세

| Tool | 이름 | 설명 |
|------|------|------|
| **Tool 1** | **Upstage Document Parser** | 업로드된 문서를 구조화 |
| **Tool 2** | **RAG Validator** | ICH/MFDS 문서를 기반으로 근거 검증 및 보완 피드백 제공 |
| **Tool 3** | **LLM CTD Writer** | Solar Pro 2 / Llama-3.2-1B-Instruct 모델 기반 CTD 문서 자동 작성 |

---

## 🔁 검증 피드백 루프

1. 입력 파일을 LLM Router가 분류 (온디바이스)  
2. Tool 1~3 처리 (보조 분기 포함)  
3. YAML 규칙 및 RAG 결과로 검증 수행  
4. 부족한 근거나 누락 발견 시 루프 재진입  
5. 최종 CTD 문서 출력  

---

## 📦 향후 계획

- 🔍 ICH/MFDS 문서 자동 크롤링 및 인용형 업데이트  
- 🧩 사용자 정의 YAML 규칙 관리 도구  
- 💾 온디바이스 최적화  
- 🧠 자동 용어 매칭 + 다국어 Translation Layer 추가  

---

---

## 요구사항
- Python 3.12.11
- OS: Ubuntu Linux 22.04

## 설치
```bash
git clone https://github.com/krnooby/CTDMate.git
cd CTDMATE
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env .env.example
# CTDMate
