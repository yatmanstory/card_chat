# eval.py — Model Based Evaluation (Synced with app.py)
import json
import os
import re
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VECTOR_STORE_DIR = "./VectorStores_Card"

# ── 1. app.py와 동일한 Retriever 파이프라인 초기화 ───────────────────────────
print("Retriever 초기화 중 (app.py 로직 동기화)...")
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)

_raw = vectordb.get()
_all_docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(_raw["documents"], _raw["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(_all_docs, k=15)
base_retriever = vectordb.as_retriever(search_kwargs={"k": 15})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, base_retriever],
    weights=[0.4, 0.6],
)
# app.py와 동일한 rerank-v3.5 사용
my_rerank = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-v3.5", top_n=6)
cohere_retriever = ContextualCompressionRetriever(
    base_compressor=my_rerank, base_retriever=ensemble_retriever
)
print(f"Retriever 초기화 완료 ({len(_all_docs)}개 문서 로드됨)")

# ── 2. 설정 및 테스트 케이스 ──────────────────────────────────────────────
BASE_MODEL = "gpt-3.5-turbo-16k"  # app.py 모델과 일치
COMPARISON_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-5-mini"
NUM_RUNS = 3  # 테스트 속도를 위해 조정 (필요시 늘리세요)
OUTPUT_DIR = "eval_results"

TEST_CASES = [
    {
        "name": "20대 김서연 대학생",
        "pattern": "서울 대학 재학 중으로 용돈과 아르바이트 수입으로 월 80만원 정도 소비/ 대중교통 이용 비용은 월 6만원이며 카페 이용 12만원, 배달 음식 15만원, 편의점 소비 10만원 정도 넷플릭스와 유튜브 프리미엄 등 구독 서비스를 이용하고 있고 소액 결제 할인과 교통, 카페, 배달 할인 혜택이 필요하다.",
    },
    {
        "name": "40대 최성호 가장",
        "pattern": "부산 거주 40대 직장인으로 가족과 함께 생활하며 월 생활비 약 250만원을 소비/ 대형마트와 식료품 구매에 월 80만원, 주유비 월 25만원, 통신비 15만원, 병원비 월 10만원 정도를 사용, 가족 중심 소비가 많아 생활비 절약과 주유, 마트 할인 혜택이 필요하다.",
    },
    {
        "name": "30대 정유진 프리랜서 디자이너",
        "pattern": "서울 거주 프리랜서 디자이너로 월 평균 수입 400만원 해외여행을 연 3회 이상 다니며 연간 여행 경비는 약 500만원 항공권과 호텔 예약, 해외 결제가 많고, 명품 및 온라인 쇼핑에 월 40만원, 카페 및 외식에 월 30만원을 소비 마일리지 적립, 해외결제 수수료 할인, 공항 라운지 혜택이 필요하다.",
    },
    {
        "name": "50대 맞벌이 부부 김현우·이수진",
        "pattern": "부산 거주 맞벌이 부부로 월 합산 실수령 약 650만원 자녀가 대학에 재학 중으로 등록금 및 교육비로 연 800만원 이상 지출 대형마트 및 식료품 구매에 월 90만원, 공과금 및 관리비 40만원, 주유비 30만원, 통신비 20만원 정도를 사용 가족 단위 소비가 많고 교육비 부담이 커 생활비 절감과 교육비, 대형마트, 주유 할인 혜택이 필요하다.",
    },
    {
        "name": "30대 박지은 직장인",
        "pattern": "일반 직장인 월 실수령 약 300만원 부모님께 용돈으로 매달 30만원을 송금하고 있으며 본인 생활비로는 온라인 쇼핑 25만원, 카페 및 외식 20만원, 교통비 7만원 정도를 사용간편결제 및 계좌이체 이용이 많고 고정 지출과 생활 소비를 함께 관리해야 하므로 생활비 할인과 간편결제 혜택이 필요하다.",
    },
    # ... (필요한 페르소나 추가)
]


# ── 3. 프롬프트 구성 (app.py 로직 반영) ──────────────────────────────────
def extract_categories_for_filter(client, user_pattern):
    """소비 패턴에서 benefit_keywords 필터용 카테고리 추출
    app.py의 extract_consumption_pattern()과 동일한 프롬프트 로직
    """
    try:
        resp = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "사용자의 소비 패턴을 분석하여 JSON 형식으로 응답하세요.\n"
                        "반드시 'categories'라는 키에 [id, label, percent]를 포함한 리스트를 담아야 합니다.\n"
                        '예: {"categories": [{"id": "food", "label": "식비", "percent": 40}]}'
                    ),
                },
                {"role": "user", "content": user_pattern},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        clean = re.sub(r"```json|```", "", content).strip()
        return json.loads(clean).get("categories", [])
    except Exception:
        return []


def build_filtered_retriever(categories):
    """benefit_keywords 필터가 적용된 Cohere 리트리버 생성
    app.py의 search_similar_cards() benefit 필터 로직과 동일
    """
    labels = [c["label"] for c in categories if c.get("label")]
    if not labels:
        return None

    if len(labels) == 1:
        chroma_filter = {"benefit_keywords": {"$contains": labels[0]}}
    else:
        chroma_filter = {
            "$or": [{"benefit_keywords": {"$contains": lbl}} for lbl in labels]
        }

    filtered_base = vectordb.as_retriever(
        search_kwargs={"k": 15, "filter": chroma_filter}
    )
    filtered_ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, filtered_base], weights=[0.4, 0.6]
    )
    return ContextualCompressionRetriever(
        base_compressor=my_rerank, base_retriever=filtered_ensemble
    )


def get_base_recommendation(client, user_pattern):
    """Model A: app.py의 RAG 파이프라인 + 프롬프트 로직

    app.py의 main() → search_similar_cards() 흐름을 그대로 재현:
      1. extract_consumption_pattern → categories
      2. benefit_keywords 사전 필터 적용
      3. 필터 결과가 3개 미만이면 전체 검색으로 폴백
    """
    # [STEP 1] app.py와 동일: 카테고리 추출
    categories = extract_categories_for_filter(client, user_pattern)

    # [STEP 2] app.py와 동일: benefit 필터 적용 검색 → 폴백
    retriever = build_filtered_retriever(categories) if categories else None
    if retriever:
        docs = retriever.invoke(user_pattern)
        if len({d.metadata.get("card_id") for d in docs}) < 3:
            # 필터로 충분한 카드를 못 찾으면 전체 검색으로 폴백
            docs = cohere_retriever.invoke(user_pattern)
    else:
        docs = cohere_retriever.invoke(user_pattern)

    # app.py와 동일한 컨텍스트 구성 (카드별 그룹핑)
    card_contents = {}
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        card_contents.setdefault(name, []).append(doc.page_content)
    context = "\n\n---\n\n".join("\n".join(chunks) for chunks in card_contents.values())

    # app.py의 generate_chat_response 프롬프트를 평가용 JSON 구조로 소폭 변형
    sys_p = f"""당신은 카드 추천 전문가입니다.
제공된 [추천 카드 목록]을 바탕으로 아래 [사고 과정]에 따라 논리적으로 분석하여 답변하세요.

[사고 과정(Chain-of-Thought)]
1. 사용자의 질의를 분석하여 사용자의 주요 소비 패턴을(영화, 카페, 쇼핑 등)과 요구 사항을 정확하게 파악한다.
2. 카드 데이터 중 해당 요구 사항에 가장 부합하는 benefit_name을 찾는다.
3. 선정한 카드들이 사용자에게 실질적으로 어떤 이득을 주는지 논리적 근거를 정확하게 도출한다.
4. 최종적으로 서로 다른 카드 TOP 3를 선정하여 결과를 출력한다.

[답변 규칙]
- 반드시 서로 다른 카드 **TOP 3**를 순서대로 추천할 것.
- 문서에 없는 내용은 절대 지어내지 말 것.
- 답변은 아래 [예시]의 형식을 엄격히 따를 것.

[Few-Shot 예시]
질문: 30대 남성 직장인으로 일본을 좋아해서 연 2회 이상 일본 해외여행을 다님, 여행 경비는 연 300만원이고, 현지의 편의점을 자주 가는 편이야 하루 1끼는 꼭 편의점으로 해결하는 것 같아. 그리고 국내 고정 치출로는 유류비 15만원이 고정 지출이야
사고 과정: 
1. 사용자의 연령대 파악 : 30대 남성 -> 경제 활동 인구
2. 소비 패턴 : 연 2회 이상 일본 여행 다님 특정 국가 지목 "일본"
3. 주 소비처 : 여행, 경비 : 300만원
4. 부 소비처 1: 현지의 편의점 -> 일본의 편의점
5. 부 소비처 2: 국내의 유류비 
6. 종합 소비 패턴 : 여행, 일본, 주유
4. 삼성카드 & MILEAGE PLATINUM (스카이패스)의 항공 마일리지 적립, 주유 마일리지 적립 > 실질적 비용 절감 효과가 큼.

-- 출력 형식 -- 
: 1. 추천 카드: 삼성카드 & MILEAGE PLATINUM (스카이패스)
 주요 혜택: 대한 항공 마일리지 적립
 추천 사유: 모든 가맹점에서 이용금액 1,000원당 1마일리지 기본적으로 적립할 수 있어, 연 2회 이상의 여행 시 페이백 효과 큽니다.
 주요 혜택: 대한 항공 마일리지 적립
 추천 사유: SK에너지, GS칼텍스, 현대오일뱅크, S-OIL, 알뜰주유소 및 LPG충전소 등에서 1,000원당 2 마일리지 적립됨, 유류비가 고정 지출인만큼, 고정적으로 마일리지가 적립되어 여행 경비에도 유리합니다.

[추천 카드 목록]
{context}

-- 출력 형식(JSON) --
{{"recommendations": [
    {{"card_name": "카드이름", "reason": "추천 이유 설명 문장", "benefit_summary": "혜택 요약"}},
    ...
]}}"""

    try:
        resp = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_pattern},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(re.sub(r"```json|```", "", content).strip())
    except Exception as e:
        print(f"[Base] Error: {e}")
        return None


def get_comparison_recommendation(client, user_pattern):
    """Model B: gpt-4o-mini (No RAG)"""
    sys_p = """당신은 카드 추천 전문가입니다. 일반적인 지식을 바탕으로 사용자의 소비 패턴에 맞는 실제 한국 카드를 TOP 3 추천하세요.
    JSON 형식으로 'recommendations': [{'card_name': '...', 'reason': '...', 'benefit_summary': '...'}] 구조로 답변하세요."""

    try:
        resp = client.chat.completions.create(
            model=COMPARISON_MODEL,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_pattern},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        return json.loads(re.sub(r"```json|```", "", content).strip())
    except Exception as e:
        print(f"[Comparison] Error: {e}")
        return None


def judge_responses(client, user_pattern, resp_a, resp_b):
    """Judge: gpt-4o가 두 모델의 응답을 평가 (응답 정규화 로직 추가)"""

    # 판사에게 더욱 엄격한 형식을 요구하는 프롬프트
    judge_sys = """당신은 금융 상품 추천 품질 평가 전문가입니다.
    Model A(RAG 기반)와 Model B(일반 지식 기반)의 추천 결과를 비교하여 평가하세요.

    반드시 아래의 JSON 형식을 엄격히 지켜서 응답하세요. 
    'winner' 필드에는 반드시 "Model A", "Model B", "Tie" 중 하나만 문자열로 입력하세요.

    {
        "model_a_score": 1-5점(정수),
        "model_b_score": 1-5점(정수),
        "winner": "Model A" 또는 "Model B" 또는 "Tie",
        "reason": "결정적인 승리 이유 한 문장"
    }
    """

    user_msg = f"""[사용자 소비패턴]
    {user_pattern}

    [Model A (RAG 기반)]
    {json.dumps(resp_a, ensure_ascii=False)}

    [Model B (일반 지식)]
    {json.dumps(resp_b, ensure_ascii=False)}
    """

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": judge_sys},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )

        # JSON 로드
        result = json.loads(resp.choices[0].message.content)

        # --- 응답 정규화 로직 (None 방지) ---
        # 1. 키 이름이 대문자로 오거나 소문자로 오는 경우 대응
        result = {k.lower(): v for k, v in result.items()}

        winner_val = result.get("winner", "Tie")

        # 2. winner가 딕셔너리로 오는 경우 (질문자님의 사례 대응)
        if isinstance(winner_val, dict):
            # 딕셔너리 안에 'model'이나 'choice' 키가 있는지 확인
            winner_val = winner_val.get("model", winner_val.get("winner", "Tie"))

        # 3. 최종 값을 깔끔하게 정리
        if "model a" in str(winner_val).lower():
            winner_final = "Model A"
        elif "model b" in str(winner_val).lower():
            winner_final = "Model B"
        else:
            winner_final = "Tie"

        return {
            "winner": winner_final,
            "reason": result.get("reason", "N/A"),
            "score_a": result.get("model_a_score", 0),
            "score_b": result.get("model_b_score", 0),
        }

    except Exception as e:
        print(f"[Judge] 파싱 에러: {e}")
        return {"winner": "Error", "reason": str(e), "score_a": 0, "score_b": 0}


# ── 4. 메인 실행부 (출력부 수정) ──────────────────────────────────────────
def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for tc in TEST_CASES:
        print(f"\n--- 평가 시작: {tc['name']} ---")
        persona_runs = []
        for run in range(NUM_RUNS):
            print(f"Run {run+1:2d}/{NUM_RUNS}...", end=" ", flush=True)

            resp_a = get_base_recommendation(client, tc["pattern"])
            resp_b = get_comparison_recommendation(client, tc["pattern"])

            if resp_a and resp_b:
                judge_result = judge_responses(client, tc["pattern"], resp_a, resp_b)
                persona_runs.append(
                    {
                        "run": run + 1,
                        "resp_a": resp_a,
                        "resp_b": resp_b,
                        "judge": judge_result,
                    }
                )
                # 이제 judge_result['winner']는 항상 "Model A", "Model B", "Tie" 중 하나입니다.
                print(
                    f"Winner: {judge_result['winner']} (A: {judge_result['score_a']} vs B: {judge_result['score_b']})"
                )
            else:
                print("Error (Model Response Failed)")

        all_results.append({"persona": tc["name"], "runs": persona_runs})

    # 결과 저장 (report.md 생성 등 기존 로직 유지)
    with open(f"{OUTPUT_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n평가 완료. 결과가 {OUTPUT_DIR} 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
