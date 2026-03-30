"""
eval_models.py — 카드 추천 모델 비교 평가 (Streamlit)

[지원 모델]
  model_a : Cohere Rerank       (커스텀 로직 + Cohere cross-encoder)
  model_b : ChromaDB 유사도     (커스텀 로직 + 코사인 거리)
  model_c : Pure Vector Similarity (커스텀 로직 없이 원문 쿼리 → ChromaDB 코사인 거리)

[평가 구조]
  - 사이드바에서 비교할 두 모델 선택
  - TEST_CASES 5개 × NUM_RUNS 10회 = 50번 질의/모델
  - Judge: gpt-5-mini (pairwise 평가)
  - 결과: eval_results/{model_a_key}_vs_{model_b_key}.json 저장

[통계 지표]
  - 승률 (%) + Wilson 95% 신뢰구간
  - 평균 점수 ± 표준편차
  - 페르소나별 세부 승패
"""

import os
import re
import glob
import json
import math
import time
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)

# ==========================================
# 1. 전역 설정
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VECTOR_STORE_DIR = "./VectorStores_Card"
OUTPUT_DIR = "eval_results"

JUDGE_MODEL = "gpt-5-mini"
RAG_LLM_MODEL = "gpt-3.5-turbo"
PURE_LLM_MODEL = "gpt-4o-mini"  # RAG 추천 생성용 (app.py와 동일)
NUM_RUNS = 5
BATCH_WORKERS = 2  # 병렬 API 호출 수 (추천 생성 + Judge) — 429 방지용으로 낮게 유지

# 최종 결과 디렉토리 (Tab 2 · Tab 3 공통)
EVAL_FINAL_DIRS = [
    "eval_results/최종 결과/Model Based/4o_vs_4o",
    "eval_results/최종 결과/Model Based/3.5Turbo_vs_4o",
    "eval_results/최종 결과/Model Based/Cohere Vs Custom",
    "eval_results/최종 결과/Model Based/Cohere_vs_4o",
]


def _scan_eval_files(include_corrected: bool = False) -> list[str]:
    """OUTPUT_DIR + EVAL_FINAL_DIRS 에서 평가 JSON 파일 목록 반환"""
    files: list[str] = []
    for d in [OUTPUT_DIR] + EVAL_FINAL_DIRS:
        if os.path.isdir(d):
            for fp in sorted(glob.glob(f"{d}/*.json")):
                if include_corrected or "_corrected" not in fp:
                    files.append(fp)
    return files


# ==========================================
# 2. 모델 레지스트리
#    새 모델 추가 시 여기에만 등록하면 됩니다.
# ==========================================
MODEL_REGISTRY = {
    "model_a": {
        "key": "model_a",
        "name": "Model A — BM25 + Cohere Rerank",
        "short": "BM25+Cohere",
        "color": "#3b82f6",
        "description": "BM25(0.4) + Chroma Dense(0.6) Hybrid → 가중 쿼리(소비 비중 상위 카테고리 + search_profile) → Cohere cross-encoder reranking",
    },
    "model_b": {
        "key": "model_b",
        "name": "Model B — ChromaDB 멀티쿼리",
        "short": "MultiQuery",
        "color": "#10b981",
        "description": "1·2순위 키워드 교집합 Hard Filter (폴백: 합집합) + 카테고리별 멀티쿼리 RRF + 금액 바인딩 프롬프트",
    },
    "model_c": {
        "key": "model_c",
        "name": "Model C — Pure Vector Similarity",
        "short": "PureVec",
        "color": "#f59e0b",
        "description": "Based 모델 GPT-4o-mini 응답 생성",
    },
}

# ==========================================
# 3. 테스트 케이스 (eval.py 동일)
# ==========================================
TEST_CASES = [
    {
        "name": "20대 김서연 대학생",
        "pattern": "서울 소재 대학교에 재학 중이며, 용돈과 아르바이트 수입을 기반으로 생활하고 있음. 대중교통을 주요 이동 수단으로 활용하며, 카페 방문과 배달음식 이용 빈도가 높은 편이다. 넷플릭스·유튜브 프리미엄 등 구독형 서비스도 정기적으로 이용하고 있으며, 편의점 소액 결제 비중도 꾸준하다. 월 평균 지출은 약 80만원 수준으로, 소비 비중은 배달음식 15만원, 카페 12만원, 편의점 10만원, 대중교통 6만원 정도이며, 소액 결제 할인, 교통·카페·배달 할인, 구독 서비스 혜택에 대한 니즈가 높다.",
    },
    {
        "name": "40대 최성호 가장",
        "pattern": "부산 거주 40대 직장인으로 가족과 함께 생활하며 월 250만원 수준의 생활비를 지출하고 있음. 대형마트 식료품 구매와 자가용 주유 비중이 높으며, 통신비와 병원비 등 고정 지출도 꾸준히 발생한다. 가족 중심 소비 패턴이 뚜렷해 생활비 전반의 혜택 효율을 중시하는 성향이다. 월 평균 지출은 약 250만원 수준으로, 소비 비중은 대형마트/식료품 80만원, 주유비 25만원, 통신비 15만원, 병원비 10만원 정도이며, 주유 할인, 대형마트 적립·할인, 생활비 절감 혜택에 대한 니즈가 높다.",
    },
    {
        "name": "30대 정유진 프리랜서 디자이너",
        "pattern": "서울 거주 프리랜서 디자이너로 월 평균 수입 약 400만원을 유지하고 있음. 해외여행을 연 3회 이상 다니며 항공권·호텔 예약과 해외 결제 비중이 높은 라이프스타일을 갖고 있다. 온라인 쇼핑·명품 구매와 카페·외식 소비도 활발하며, 여행 관련 프리미엄 혜택에 관심이 많다. 월 평균 지출은 약 150~200만원 수준(여행 시즌 별도)으로, 소비 비중은 해외여행/항공·숙박 연 500만원, 온라인 쇼핑·명품 40만원, 카페·외식 30만원 정도이며, 마일리지 적립, 해외결제 수수료 할인, 공항 라운지 혜택에 대한 니즈가 높다.",
    },
    {
        "name": "50대 맞벌이 부부 김현우·이수진",
        "pattern": "부산 거주 맞벌이 부부로 월 합산 실수령 약 650만원을 유지하고 있음. 자녀가 대학에 재학 중으로 등록금·교육비로 연 800만원 이상을 지출하고 있으며, 대형마트 식료품 구매와 공과금·관리비, 주유비 등 고정 지출 항목이 다양하다. 가족 단위 소비가 많아 지출 항목 전반의 혜택 효율을 중시하는 편이다. 월 평균 지출은 약 400~500만원 수준으로, 소비 비중은 대형마트/식료품 90만원, 공과금·관리비 40만원, 주유비 30만원, 통신비 20만원 정도이며, 대형마트 할인, 주유비 절감, 생활비 절감 혜택에 대한 니즈가 높다.",
    },
    {
        "name": "30대 박지은 직장인",
        "pattern": "일반 직장인으로 월 실수령 약 300만원을 수령하고 있음. 매달 부모님께 30만원을 용돈으로 송금하고 있으며, 본인 생활비는 온라인 쇼핑과 카페·외식 중심으로 지출되고 있다. 평소 간편결제와 계좌이체를 적극 활용하며, 고정 지출과 생활 소비를 함께 관리해야 하는 상황이다. 월 평균 지출은 약 150~180만원 수준으로, 소비 비중은 부모님 용돈 30만원, 온라인 쇼핑 25만원, 카페·외식 20만원, 교통비 7만원 정도이며, 생활비 할인, 간편결제 캐시백·적립 혜택에 대한 니즈가 높다.",
    },
    {
        "name": "30대 김노리 직장인",
        "pattern": "테마파크 투어를 취미생활로 하고 있는 청년 월 평균 수입 300만원 중 경주월드, 에버랜드, 롯데월드 등 테마파크 투어에 월 평균 120만원 지출이 발생하고 있다.소비 비준은 식비 포함 테마파크 지출 80만원, 유류비 30만원, 숙박비 20만원 지출되고 있다.테마파크, 주유 혜택, 숙박 헤택 등 혜택이 필요하다.",
    },
    {
        "name": "60대 김피트",
        "pattern": "저명한 작곡가로 월 실수령 약 600만원을 수령하고 있음. 여유 시간에는 피트니스 즐기는 라이프스타일을 가지고 있다. 반려견을 키우고 있어 정기적인 동물병원 방문과 사료 및 용품 소비가 꾸준히 발생한다. 월 평균 지출은 약 120만원 수준으로, 소비 비중은 피트니스 70만원, 반려동물 관련 50만원,반려동물 케어 서비스, 뷰티/피트니스 관련 할인 및 적립 혜택에 대한 니즈가 높다.",
    },
    {
        "name": "10대 박열심",
        "pattern": "고등학생으로 학업에 집중하며 대학 진학을 목표로 하고 있음. 한 달 용돈과 60만원으로 주 소비처는 전공서 및 참고서 구매 비중도 높으며, 정기적인 학원비 지출이 발생한다. 월 평균 지출은 약 60만원 수준으로, 소비 비중은 학원비 25만원, 전공서/교재 10만원, 편의점 10만원, 카페/디저트 10만원 정도이며, 교육 관련 혜택, 편의점 및 카페 적립에 대한 니즈가 높다.",
    },
    {
        "name": "20대 김틱톡",
        "pattern": "SNS 기반으로 활동하는 유명 인플루언서로 광고 수익과 협찬을 통해 수입을 창출하고 있음. 콘텐츠 제작을 위해 스마트폰을 본인 명의로 3대 이상 운영하고 있어 통신비 지출이 높은 편이다. 평소 간편 결제와 소액 결제를 적극 활용하며, 라이브 커머스 및 온라인 쇼핑 플랫폼을 중심으로 소비가 이루어진다. 넷플릭스, 유튜브 프리미엄 등 온라인 구독 서비스도 이용한다. 월 평균 지출은 약 150만원 수준으로, 소비 비중은 통신비 100만원, 카페 및 외식 30만원 정도이며, OTT/구독 서비스 유지에 20만원을 사용한다. 간편결제 할인, 통신비 절감 혜택, 구독형 서비스 할인에 대한 니즈가 높다.",
    },
    {
        "name": "30대 피우리",
        "pattern": "30대 미혼모로 3세 자녀를 양육하며 생계와 육아를 병행하고 있음. 자녀의 건강 상태로 인해 병원과 약국 이용 빈도가 높으며, 예방접종 및 정기 진료 등 의료비 지출 비중이 큰 편이다. 의료·보건 바우처와 정부 지원 정책에 대한 관심이 높고, 이를 적극적으로 활용하려는 성향을 보인다. 육아로 인한 스트레스 관리와 자기 관리를 위해 뷰티 및 피트니스 소비도 일정 비중 유지하고 있다. 월 평균 지출은 약 140만원 수준으로, 소비 비중은 병원/약국 100만원, 뷰티/피트니스 40만원 정도이며, 의료비 할인 및 적립, 약국/병원 제휴 혜택, 정부 바우처 연계 서비스, 육아 및 헬스케어 관련 혜택에 대한 니즈가 높다.",
    },
]


# ==========================================
# 4. 리소스 초기화 (캐시)
# ==========================================
@st.cache_resource
def init_resources():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)

    rag_llm = ChatOpenAI(model=RAG_LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    pure_vec_llm = ChatOpenAI(
        model=PURE_LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY
    )

    my_rerank = CohereRerank(
        cohere_api_key=COHERE_API_KEY, model="rerank-v3.5", top_n=6
    )
    openai_judge = OpenAI(api_key=OPENAI_API_KEY)

    _chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    raw_collection = _chroma_client.get_collection(
        _chroma_client.list_collections()[0].name
    )

    with open("data/categories_rows.json", "r", encoding="utf-8") as f:
        CATEGORIES = json.load(f)
    CATEGORY_MAP = {c["name"]: json.loads(c["mapped_names"]) for c in CATEGORIES}

    # BM25 + Chroma Hybrid → Cohere (Model A용)
    _raw = vectordb.get()
    _all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(_raw["documents"], _raw["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(_all_docs, k=15)
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, base_retriever], weights=[0.3, 0.7]
    )
    cohere_retriever = ContextualCompressionRetriever(
        base_compressor=my_rerank, base_retriever=ensemble_retriever
    )

    return {
        "vectordb": vectordb,
        "rag_llm": rag_llm,
        "pure_vec_llm": pure_vec_llm,
        "my_rerank": my_rerank,
        "openai_judge": openai_judge,
        "raw_collection": raw_collection,
        "CATEGORIES": CATEGORIES,
        "CATEGORY_MAP": CATEGORY_MAP,
        "cohere_retriever": cohere_retriever,
    }


# ==========================================
# 5. 공통 파이프라인 유틸
# ==========================================

# 소비 패턴 추출 결과 캐시 — 동일 페르소나는 1회만 LLM 호출
_extraction_cache: dict[str, dict] = {}


def extract_consumption_pattern(user_text, rag_llm, CATEGORIES):
    if user_text in _extraction_cache:
        return _extraction_cache[user_text]

    category_list = "\n".join(f'- id={c["id"]}, name="{c["name"]}"' for c in CATEGORIES)
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """**사용자의 소비 패턴을 분석**하여 JSON 형식으로 응답하세요.
반드시 'categories'와 'search_profile' 두 가지 키를 포함해야 합니다.

1. 'categories': 사용자 질의에서 가장 지출 비중이 높은 핵심 분야만 **최대 4개** 선정하여 [id, name, percent] 리스트로 담으세요.
2. 선정한 항목의 percent 합계는 반드시 100이 되어야 합니다.
3. 'search_profile': 사용자의 상황과 필요한 혜택을 모든 키워드를 포함해 상세한 문장으로 재작성하세요.

[허용 카테고리 목록]:
{category_list}

응답 예시:
{{
    "categories": [
        {{"id": 3, "name": "카페/식음료", "percent": 30}},
        {{"id": 4, "name": "배달앱", "percent": 40}}
    ],
    "search_profile": "20대 대학생으로 배달음식과 카페 이용 비중이 높으며 교통비와 OTT 구독 할인이 필요한 상황"
}}
""",
            ),
            ("human", "{question}"),
        ]
    )
    chain = extract_prompt | rag_llm
    response = chain.invoke({"question": user_text, "category_list": category_list})
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_json)
        if "categories" in data:
            data["categories"] = sorted(
                data["categories"], key=lambda x: x.get("percent", 0), reverse=True
            )[:3]
        _extraction_cache[user_text] = data
        return data
    except (json.JSONDecodeError, KeyError):
        fallback = {"categories": [], "search_profile": user_text}
        _extraction_cache[user_text] = fallback
        return fallback


def get_search_keywords(categories, CATEGORY_MAP):
    keywords = []
    for cat in categories:
        mapped = CATEGORY_MAP.get(cat.get("name", ""), [cat.get("name", "")])
        keywords.extend(mapped)
    return list(set(keywords))


def _build_keyword_filter(keywords):
    if not keywords:
        return None
    if len(keywords) == 1:
        return {"benefit_keywords": {"$contains": keywords[0]}}
    return {"$or": [{"benefit_keywords": {"$contains": kw}} for kw in keywords]}


def _get_card_match_count(raw_collection, keywords):
    """키워드 목록에 매칭되는 card_id별 청크 수 반환"""
    if not keywords:
        return {}
    chroma_filter = _build_keyword_filter(keywords)
    results = raw_collection.get(where=chroma_filter, include=["metadatas"])
    count = {}
    for meta in results["metadatas"]:
        cid = meta.get("card_id", "")
        if cid:
            count[cid] = count.get(cid, 0) + 1
    return count


def _build_id_filter(top_card_ids):
    if len(top_card_ids) == 1:
        return {"card_id": {"$eq": top_card_ids[0]}}
    return {"$or": [{"card_id": {"$eq": cid}} for cid in top_card_ids]}


def _get_card_names_from_docs(docs, top_k=3):
    seen, names = set(), []
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) >= top_k:
            break
    return names


def _filter_docs_to_top_cards(docs, top_k=3):
    """app.py와 동일: rerank 결과에서 상위 top_k 카드 card_id만 추출 후
    해당 카드의 청크만 반환 → 토큰 초과 방지"""
    seen, card_ids = set(), []
    for doc in docs:
        cid = doc.metadata.get("card_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            card_ids.append(cid)
        if len(card_ids) >= top_k:
            break
    top_ids = set(card_ids)
    return [d for d in docs if d.metadata.get("card_id") in top_ids]


def _build_rag_prompt(user_text, docs, categories):
    pattern_summary = ", ".join([f"{c['name']}({c['percent']}%)" for c in categories])
    card_contents: dict[str, list[str]] = {}
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        cleaned = re.sub(
            r"\[유의사항\].*?(?=\n\[|\Z)", "", doc.page_content, flags=re.DOTALL
        ).strip()
        if cleaned:
            card_contents.setdefault(name, []).append(cleaned)

    context = "\n\n---\n\n".join(
        f"【 카드명: {n} 】\n" + "\n\n".join(chunks)
        for n, chunks in card_contents.items()
    )
    card_names = list(card_contents.keys())
    cn1 = card_names[0] if len(card_names) > 0 else "카드 1"
    cn2 = card_names[1] if len(card_names) > 1 else "카드 2"
    cn3 = card_names[2] if len(card_names) > 2 else "카드 3"

    return f"""당신은 카드 추천 전문가입니다. 사용자의 주요 소비 패턴: {pattern_summary}

[추천 카드 목록]
{context}

[사용자 질문]
{user_text}

[답변 규칙]
1. 반드시 {cn1}, {cn2}, {cn3} 세 카드를 모두 순서대로 추천하세요.
2. 각 카드별로 사용자가 직접 언급한 소비 패턴과 가장 관련된 혜택 2가지를 선정하세요.
3. 카드명, 혜택명1, 혜택명2, 추천 이유는 반드시 각각 새로운 줄에서 시작하세요.
4. 추천 이유는 사용자가 언급한 구체적인 수치와 카드 혜택 조건을 연결하여 2~3문장으로 작성하세요.
5. 문서에 없는 내용은 절대 지어내지 마세요.

-- 출력 형식 --
#### [**{cn1}**]
**혜택명1**: 혜택 요약 1줄
**혜택명2**: 혜택 요약 1줄
- 추천 이유: 2~3문장

#### [**{cn2}**]
**혜택명1**: 혜택 요약 1줄
**혜택명2**: 혜택 요약 1줄
- 추천 이유: 2~3문장

#### [**{cn3}**]
**혜택명1**: 혜택 요약 1줄
**혜택명2**: 혜택 요약 1줄
- 추천 이유: 2~3문장"""


def _build_rag_prompt_b(user_text, docs, categories):
    """Model B 전용 프롬프트: 금액 바인딩 강제 + 소비 비중 우선순위 명시"""
    pattern_summary = ", ".join([f"{c['name']}({c['percent']}%)" for c in categories])
    card_contents: dict[str, list[str]] = {}
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        cleaned = re.sub(
            r"\[유의사항\].*?(?=\n\[|\Z)", "", doc.page_content, flags=re.DOTALL
        ).strip()
        if cleaned:
            card_contents.setdefault(name, []).append(cleaned)

    context = "\n\n---\n\n".join(
        f"【 카드명: {n} 】\n" + "\n\n".join(chunks)
        for n, chunks in card_contents.items()
    )
    card_names = list(card_contents.keys())
    cn1 = card_names[0] if len(card_names) > 0 else "카드 1"
    cn2 = card_names[1] if len(card_names) > 1 else "카드 2"
    cn3 = card_names[2] if len(card_names) > 2 else "카드 3"

    # 소비 비중 순위 텍스트 생성
    priority_text = " > ".join(f"{c['name']}({c['percent']}%)" for c in categories)

    return f"""당신은 카드 추천 전문가입니다.
사용자의 소비 비중 우선순위: {priority_text}
비중이 높은 항목일수록 해당 혜택이 강한 카드를 우선 추천하세요.

[추천 카드 목록]
{context}

[사용자 질문]
{user_text}

[답변 규칙]
1. 반드시 {cn1}, {cn2}, {cn3} 세 카드를 모두 순서대로 추천하세요.
2. 각 카드별로 소비 비중 상위 항목과 직접 연관된 혜택을 우선 선정하세요.
3. 카드명, 혜택명1, 혜택명2, 추천 이유는 반드시 각각 새로운 줄에서 시작하세요.
4. [금액 바인딩 필수] 사용자가 언급한 구체적 금액(예: "교통비 월 7만원")과 카드의 실제 혜택 조건(예: "대중교통 월 최대 5,000원 할인, 전월실적 30만원 이상")을 반드시 명시적으로 연결하세요.
   좋은 예: "월 15만원 배달 지출 기준, 이 카드의 배달앱 10% 할인 적용 시 월 최대 1만 5천원 절약 가능합니다."
   나쁜 예: "배달앱 할인 혜택이 있어 배달을 자주 이용하는 분께 적합합니다." (금액 연결 없는 일반적 설명 금지)
5. 문서에 없는 수치나 혜택은 절대 지어내지 마세요.

-- 출력 형식 --
#### [**{cn1}**]
**혜택명1**: 혜택 요약 1줄 (실제 조건 포함)
**혜택명2**: 혜택 요약 1줄 (실제 조건 포함)
- 추천 이유: 사용자 금액 ↔ 카드 조건 연결 2~3문장

#### [**{cn2}**]
**혜택명1**: 혜택 요약 1줄 (실제 조건 포함)
**혜택명2**: 혜택 요약 1줄 (실제 조건 포함)
- 추천 이유: 사용자 금액 ↔ 카드 조건 연결 2~3문장

#### [**{cn3}**]
**혜택명1**: 혜택 요약 1줄 (실제 조건 포함)
**혜택명2**: 혜택 요약 1줄 (실제 조건 포함)
- 추천 이유: 사용자 금액 ↔ 카드 조건 연결 2~3문장"""


# ==========================================
# 6. 모델별 추천 생성 함수
#    새 모델 추가 시 이 섹션에 함수 추가 후
#    get_recommendation()의 dispatch 딕셔너리에 등록
# ==========================================
def _recommend_model_a(user_text, resources):
    """Model A: BM25 + Chroma Hybrid → Cohere Rerank (커스텀 필터 없음)"""
    analysis = extract_consumption_pattern(
        user_text, resources["rag_llm"], resources["CATEGORIES"]
    )
    categories = analysis.get("categories", [])
    search_profile = analysis.get("search_profile", user_text)

    # 소비 비중 상위 2개 카테고리 + 전체 소비 프로필로 가중 쿼리 구성
    top_names = [c["name"] for c in categories[:2] if c.get("name")]
    if top_names:
        weighted_query = f"[{', '.join(top_names)} 집중 할인] {search_profile}"
    else:
        weighted_query = search_profile

    docs = resources["cohere_retriever"].invoke(weighted_query)
    filtered_docs = _filter_docs_to_top_cards(docs, top_k=3)
    prompt = _build_rag_prompt(user_text, filtered_docs, categories)
    response = resources["rag_llm"].invoke(prompt)
    return response.content, _get_card_names_from_docs(filtered_docs), {}


def _recommend_model_b(user_text, resources):
    """Model B: 교집합 Hard Filter + 멀티쿼리 RRF + 금액 바인딩 프롬프트
    - 1·2순위 카테고리 키워드 교집합 우선 → 부족 시 합집합 폴백
    - 카테고리별 비중 강조 쿼리 + 통합 가중 쿼리로 멀티쿼리 구성
    - percent 가중 RRF로 card_id 스코어 합산 → top-3 선별
    - _build_rag_prompt_b로 사용자 금액 ↔ 카드 조건 명시적 연결
    """
    analysis = extract_consumption_pattern(
        user_text, resources["rag_llm"], resources["CATEGORIES"]
    )
    categories = analysis.get("categories", [])
    search_profile = analysis.get("search_profile", "")

    # 후보 card_id 풀 결정: 1·2순위 교집합 우선 → 합집합 폴백
    top1_keywords = get_search_keywords(categories[:1], resources["CATEGORY_MAP"])
    top2_keywords = (
        get_search_keywords(categories[1:2], resources["CATEGORY_MAP"])
        if len(categories) > 1
        else []
    )
    all_keywords = get_search_keywords(categories, resources["CATEGORY_MAP"])
    raw = resources["raw_collection"]
    id_filter = None
    card_pool = 9
    top_k = 3

    candidate_mode = "none"
    intersection_size = 0

    if top1_keywords and top2_keywords:
        top1_count = _get_card_match_count(raw, top1_keywords)
        top2_count = _get_card_match_count(raw, top2_keywords)
        intersection_ids = set(top1_count) & set(top2_count)
        intersection_size = len(intersection_ids)

        if intersection_size >= top_k:
            all_count = _get_card_match_count(raw, all_keywords)
            top_ids = sorted(
                intersection_ids,
                key=lambda x: all_count.get(x, 0),
                reverse=True,
            )[:card_pool]
            candidate_mode = "intersection"
        else:
            all_count = _get_card_match_count(raw, all_keywords)
            top_ids = sorted(all_count, key=lambda x: all_count[x], reverse=True)[
                :card_pool
            ]
            candidate_mode = "union"
    elif all_keywords:
        all_count = _get_card_match_count(raw, all_keywords)
        top_ids = sorted(all_count, key=lambda x: all_count[x], reverse=True)[
            :card_pool
        ]
        candidate_mode = "union"
    else:
        top_ids = []

    if len(top_ids) >= top_k:
        id_filter = _build_id_filter(top_ids)

    # 멀티쿼리 구성: 비중 순위 명시 카테고리별 쿼리 + 상위 카테고리 강조 통합 쿼리
    rank_labels = ["1순위", "2순위", "3순위", "4순위"]
    queries: list[tuple[str, float]] = [
        (
            f"[{rank_labels[i] if i < 4 else f'{i+1}순위'} {cat['name']} {cat['percent']}%] "
            f"{cat['name']} 집중 할인 혜택 카드",
            cat["percent"],
        )
        for i, cat in enumerate(categories)
    ]
    # 상위 2개 카테고리를 강조한 통합 프로필 쿼리
    top_cats = categories[:2]
    top_emphasis = ", ".join(
        f"{c['name']}({c['percent']}%) 집중 할인" for c in top_cats
    )
    top_percent = max((c["percent"] for c in categories), default=50)
    integrated_query = (
        f"{top_emphasis}. {search_profile}. " f"{top_cats[0]['name']} 할인이 가장 중요."
        if top_cats
        else search_profile
    )
    queries.append((integrated_query, top_percent))

    # card_id별 RRF 스코어 집계 (score = percent / (rank + 1))
    MAX_CHUNKS_PER_CARD = 10  # LLM 컨텍스트 토큰 초과 방지
    card_score: dict[str, float] = {}
    card_docs: dict[str, list] = {}
    card_seen_contents: dict[str, set] = {}  # 카드별 중복 청크 방지

    for query_text, weight in queries:
        search_k = 9 if id_filter else 15
        docs = resources["vectordb"].similarity_search(
            query_text, k=search_k, **({"filter": id_filter} if id_filter else {})
        )
        seen: set[str] = set()
        for rank, doc in enumerate(docs):
            cid = doc.metadata.get("card_id", "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            card_score[cid] = card_score.get(cid, 0) + weight / (rank + 1)
            # 중복 청크 제거 + 카드당 최대 MAX_CHUNKS_PER_CARD개 보존
            existing = card_seen_contents.setdefault(cid, set())
            if (
                doc.page_content not in existing
                and len(card_docs.get(cid, [])) < MAX_CHUNKS_PER_CARD
            ):
                existing.add(doc.page_content)
                card_docs.setdefault(cid, []).append(doc)

    # 스코어 상위 top-3 card_id 선별
    top_card_ids = sorted(card_score, key=lambda x: card_score[x], reverse=True)[:3]

    # 해당 카드 docs만 모아서 LLM 컨텍스트 구성
    filtered_docs = [
        doc
        for cid in top_card_ids
        for doc in card_docs.get(cid, [])
        if doc.metadata.get("card_id") == cid
    ]

    prompt = _build_rag_prompt_b(user_text, filtered_docs, categories)
    response = resources["rag_llm"].invoke(prompt)
    debug_info = {
        "candidate_mode": candidate_mode,
        "intersection_size": intersection_size,
        "pool_size": len(top_ids),
    }
    return response.content, _get_card_names_from_docs(filtered_docs), debug_info


def _recommend_model_c(user_text, resources):
    """Model C: Pure Vector Similarity — 사용자 원문을 그대로 ChromaDB에 쿼리"""
    docs = resources["vectordb"].similarity_search(user_text, k=9)
    filtered_docs = _filter_docs_to_top_cards(docs, top_k=3)
    card_names = _get_card_names_from_docs(filtered_docs)

    # categories 없이 동작하는 단순 프롬프트
    card_contents: dict[str, list[str]] = {}
    for doc in filtered_docs:
        name = doc.metadata.get("card_name", "")
        cleaned = re.sub(
            r"\[유의사항\].*?(?=\n\[|\Z)", "", doc.page_content, flags=re.DOTALL
        ).strip()
        if cleaned:
            card_contents.setdefault(name, []).append(cleaned)

    context = "\n\n---\n\n".join(
        f"【 카드명: {n} 】\n" + "\n\n".join(chunks)
        for n, chunks in card_contents.items()
    )
    cn1 = card_names[0] if len(card_names) > 0 else "카드 1"
    cn2 = card_names[1] if len(card_names) > 1 else "카드 2"
    cn3 = card_names[2] if len(card_names) > 2 else "카드 3"

    prompt = f"""당신은 카드 추천 전문가입니다.
아래 [추천 카드 목록]을 기준으로 사용자 질문에 맞는 혜택을 설명하세요.

[추천 카드 목록]
{context}

[사용자 질문]
{user_text}

[답변 규칙]
1. 반드시 {cn1}, {cn2}, {cn3} 세 카드를 모두 순서대로 추천하세요.
2. 각 카드별로 사용자 질문과 관련된 혜택 2가지를 선정하세요.
3. 추천 이유는 카드 혜택과 사용자 질문을 연결하여 2~3문장으로 작성하세요.
4. 문서에 없는 내용은 절대 지어내지 마세요.

-- 출력 형식 --
#### [**{cn1}**]
**혜택명1**: 혜택 요약 1줄
**혜택명2**: 혜택 요약 1줄
- 추천 이유: 2~3문장

#### [**{cn2}**]
**혜택명1**: 혜택 요약 1줄
**혜택명2**: 혜택 요약 1줄
- 추천 이유: 2~3문장

#### [**{cn3}**]
**혜택명1**: 혜택 요약 1줄
**혜택명2**: 혜택 요약 1줄
- 추천 이유: 2~3문장"""

    response = resources["pure_vec_llm"].invoke(prompt)
    return response.content, card_names, {}


# 모델 키 → 추천 함수 디스패치 테이블
# 새 모델 추가 시 여기에 등록
_RECOMMEND_DISPATCH = {
    "model_a": _recommend_model_a,
    "model_b": _recommend_model_b,
    "model_c": _recommend_model_c,
}


def get_recommendation(model_key: str, user_text: str, resources: dict):
    """모델 키로 추천 함수를 호출하는 단일 진입점"""
    fn = _RECOMMEND_DISPATCH.get(model_key)
    if fn is None:
        raise ValueError(f"알 수 없는 모델 키: {model_key}")
    return fn(user_text, resources)


# ==========================================
# 7. Judge (gpt-5-mini)
# ==========================================
def judge_responses(openai_judge, user_pattern, key_x, resp_x, key_y, resp_y):
    name_x = MODEL_REGISTRY[key_x]["name"]
    name_y = MODEL_REGISTRY[key_y]["name"]

    judge_sys = f"""당신은 금융 상품 추천 품질 평가 전문가입니다.
두 모델의 카드 추천 결과를 아래 기준으로 평가하세요.

[평가 기준]
1. 추천 정확도: 사용자 소비 패턴과 **카드 혜택의 실제 일치도** (환각 혜택 설명 감점)
2. 논리성: 추천 이유가 사용자의 구체적 수치와 연결되는지
3. 관련성: 추천 카드가 사용자 요구에 얼마나 부합하는지

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "score_x": 1~5 정수,
    "score_y": 1~5 정수,
    "winner": "{name_x}" 또는 "{name_y}" 또는 "Tie",
    "reason": "결정적인 판정 이유 한 문장"
}}"""

    user_msg = f"""[사용자 소비 패턴]
{user_pattern}

[{name_x}]
{resp_x}

[{name_y}]
{resp_y}"""

    try:
        resp = openai_judge.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": judge_sys},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result = {k.lower(): v for k, v in result.items()}

        winner_raw = str(result.get("winner", "Tie"))
        if MODEL_REGISTRY[key_x]["name"].lower() in winner_raw.lower():
            winner = key_x
        elif MODEL_REGISTRY[key_y]["name"].lower() in winner_raw.lower():
            winner = key_y
        else:
            winner = "tie"

        return {
            "winner": winner,
            "reason": result.get("reason", "N/A"),
            "score_x": int(result.get("score_x", 0)),
            "score_y": int(result.get("score_y", 0)),
        }
    except Exception as e:
        return {"winner": "error", "reason": str(e), "score_x": 0, "score_y": 0}


# ==========================================
# 7-2. 혜택 정확도 검증 Judge (gpt-5-mini 재사용)
# ==========================================
def _sanitize(text, maxlen: int = 3000) -> str:
    """LLM 응답에 섞일 수 있는 제어 문자(U+0000~U+001F, \\x00 등)를 제거하고 길이를 제한.
    JSON body 직렬화 실패(400 error)를 방지한다."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    # \\n(0x0A) · \\t(0x09) · \\r(0x0D) 은 허용, 나머지 C0 제어 문자 제거
    cleaned = "".join(c for c in text if ord(c) >= 0x20 or c in "\n\t\r")
    return cleaned[:maxlen]


def verify_benefits_accuracy(
    openai_judge, user_pattern, card_names, cards_lookup, model_response, model_label
):
    """RAG 데이터를 기반으로 모델 응답의 카드 혜택 정확도를 gpt-5-mini가 검증한다.

    Args:
        cards_lookup: {card_name: benefits_text} — cards.json에서 구성
    Returns:
        dict with keys: accuracy_score(1-5), hallucination_detected(bool), issues(str), verdict("PASS"/"FAIL"/"SKIP"/"ERROR")
    """
    # 실제 카드 데이터 컨텍스트 구성
    card_context_parts = []
    for name in card_names:
        if name in cards_lookup:
            card_context_parts.append(f"【 {name} 실제 혜택 】\n{cards_lookup[name]}")
    if not card_context_parts:
        return {
            "accuracy_score": 0,
            "hallucination_detected": False,
            "issues": "cards.json에 해당 카드 없음 — 검증 불가",
            "verdict": "SKIP",
        }

    card_context = "\n\n".join(card_context_parts)
    user_pattern = _sanitize(user_pattern, 600)
    card_context = _sanitize(card_context, 2000)
    model_response = _sanitize(model_response, 3000)

    sys_prompt = """당신은 신용카드 혜택 정확성 검증 전문가입니다.
RAG 데이터베이스의 실제 카드 혜택 정보를 기반으로, AI 모델이 추천한 카드의 혜택 설명이 정확한지 평가하세요.

[평가 기준]
1. 혜택 존재 여부: 실제 카드 데이터에 해당 혜택이 존재하는가
2. 혜택 수치 정확성: 할인율, 한도, 적립률 등 수치가 카드 데이터와 일치하는가
3. 혜택 조건 정확성: 전월실적 조건, 월 한도 등 적용 조건이 올바른가
4. 환각 감지: 카드 데이터에 없는 허위 혜택이나 수치를 생성했는가

반드시 아래 JSON 형식으로만 응답하세요:
{
    "accuracy_score": 1~5 정수 (5=완전히 정확, 1=대부분 부정확/환각),
    "hallucination_detected": true 또는 false,
    "issues": "발견된 부정확한 내용 (없으면 '없음')",
    "verdict": "PASS" 또는 "FAIL"
}"""

    user_msg = f"""[사용자 소비 패턴]
{user_pattern}

[실제 카드 데이터 (RAG DB)]
{card_context}

[{model_label}의 추천 응답]
{model_response}"""

    try:
        resp = openai_judge.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result = {k.lower(): v for k, v in result.items()}
        return result
    except Exception as e:
        return {
            "accuracy_score": 0,
            "hallucination_detected": False,
            "issues": str(e),
            "verdict": "ERROR",
        }


# ==========================================
# 8. 평가 실행
# ==========================================
def run_evaluation(key_x, key_y, resources):
    """
    2-phase 병렬 평가
      Phase 1 — 추천 생성: BATCH_WORKERS 동시 호출 (소비 패턴 추출은 캐시)
      Phase 2 — Judge:     BATCH_WORKERS 동시 호출
    총 API 호출 수 절감:
      - extract_consumption_pattern: NUM_RUNS × 2 → 1 × 2 (페르소나당)
      - 추천 + Judge: 순차 대비 ~BATCH_WORKERS배 빠름
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/{key_x}_vs_{key_y}.json"

    total = len(TEST_CASES) * NUM_RUNS
    progress_bar = st.progress(0, text="평가 준비 중...")
    status = st.empty()

    all_tasks = [(tc, run) for tc in TEST_CASES for run in range(NUM_RUNS)]
    rec_results: dict[tuple, dict] = {}
    judge_results: dict[tuple, dict] = {}

    # ── Phase 1: 추천 생성 (병렬) ──────────────────────────────────────
    def _gen_pair(args):
        tc, run_idx = args
        max_retries = 6
        for attempt in range(max_retries):
            try:
                resp_x, cards_x, debug_x = get_recommendation(
                    key_x, tc["pattern"], resources
                )
                resp_y, cards_y, debug_y = get_recommendation(
                    key_y, tc["pattern"], resources
                )
                return (tc["name"], run_idx), {
                    "resp_x": resp_x,
                    "cards_x": cards_x,
                    "debug_x": debug_x,
                    "resp_y": resp_y,
                    "cards_y": cards_y,
                    "debug_y": debug_y,
                    "error": None,
                }
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2**attempt + 2  # 3, 4, 6, 10, 18초 대기
                    time.sleep(wait)
                else:
                    return (tc["name"], run_idx), {"error": str(e)}

    status.markdown("**[1/2] 추천 생성 중 (병렬)...**")
    done = 0
    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as pool:
        futures = {pool.submit(_gen_pair, task): task for task in all_tasks}
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(done / (total * 2), text=f"추천 생성 {done}/{total}")
            task_key, val = future.result()
            rec_results[task_key] = val

    # ── Phase 2: Judge (병렬) ────────────────────────────────────────────
    def _judge(args):
        tc, run_idx = args
        rec = rec_results.get((tc["name"], run_idx), {})
        if rec.get("error"):
            return (tc["name"], run_idx), {
                "winner": "error",
                "reason": rec["error"],
                "score_x": 0,
                "score_y": 0,
            }
        max_retries = 6
        for attempt in range(max_retries):
            try:
                result = judge_responses(
                    resources["openai_judge"],
                    tc["pattern"],
                    key_x,
                    rec["resp_x"],
                    key_y,
                    rec["resp_y"],
                )
                return (tc["name"], run_idx), result
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2**attempt + 2  # 3, 4, 6, 10, 18초 대기
                    time.sleep(wait)
                else:
                    return (tc["name"], run_idx), {
                        "winner": "error",
                        "reason": str(e),
                        "score_x": 0,
                        "score_y": 0,
                    }

    status.markdown("**[2/2] 판정 중 (병렬)...**")
    done = 0
    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as pool:
        futures = {pool.submit(_judge, task): task for task in all_tasks}
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(0.5 + done / (total * 2), text=f"판정 {done}/{total}")
            task_key, val = future.result()
            judge_results[task_key] = val

    # ── 결과 조립 ────────────────────────────────────────────────────────
    all_results = []
    for tc in TEST_CASES:
        persona_runs = []
        for run_idx in range(NUM_RUNS):
            rec = rec_results.get((tc["name"], run_idx), {"error": "missing"})
            judge = judge_results.get(
                (tc["name"], run_idx),
                {"winner": "error", "reason": "missing", "score_x": 0, "score_y": 0},
            )
            run_data: dict = {"run": run_idx + 1, "judge": judge}
            if rec.get("error"):
                run_data["error"] = rec["error"]
            else:
                run_data.update(
                    {
                        "resp_x": rec["resp_x"],
                        "cards_x": rec["cards_x"],
                        "debug_x": rec.get("debug_x", {}),
                        "resp_y": rec["resp_y"],
                        "cards_y": rec["cards_y"],
                        "debug_y": rec.get("debug_y", {}),
                    }
                )
            persona_runs.append(run_data)
        all_results.append({"persona": tc["name"], "runs": persona_runs})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"key_x": key_x, "key_y": key_y, "results": all_results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    progress_bar.progress(1.0, text="평가 완료!")
    status.success(f"저장 완료: `{output_file}`")

    # Model B 후보 풀 분석 요약
    b_key = key_y if key_y == "model_b" else (key_x if key_x == "model_b" else None)
    if b_key:
        debug_field = "debug_y" if b_key == key_y else "debug_x"
        rows_debug = []
        for tc_result in all_results:
            for r in tc_result["runs"]:
                d = r.get(debug_field, {})
                if d:
                    rows_debug.append(
                        {
                            "persona": tc_result["persona"],
                            "run": r["run"],
                            "candidate_mode": d.get("candidate_mode", "-"),
                            "intersection_size": d.get("intersection_size", 0),
                            "pool_size": d.get("pool_size", 0),
                        }
                    )
        if rows_debug:
            df_debug = pd.DataFrame(rows_debug)
            total_runs = len(df_debug)
            n_intersection = (df_debug["candidate_mode"] == "intersection").sum()
            n_union = (df_debug["candidate_mode"] == "union").sum()
            with st.expander("🔍 Model B 후보 풀 분석 (intersection vs union)"):
                c1, c2, c3 = st.columns(3)
                c1.metric("전체 실행", total_runs)
                c2.metric(
                    "교집합 사용",
                    f"{n_intersection}회 ({n_intersection/total_runs*100:.0f}%)",
                )
                c3.metric("합집합 폴백", f"{n_union}회 ({n_union/total_runs*100:.0f}%)")
                st.dataframe(
                    df_debug.groupby(["persona", "candidate_mode"])
                    .size()
                    .reset_index(name="count")
                    .pivot(index="persona", columns="candidate_mode", values="count")
                    .fillna(0)
                    .astype(int),
                    use_container_width=True,
                )

    return all_results, key_x, key_y


# ==========================================
# 9. 통계 계산
# ==========================================
def wilson_ci(wins, n, z=1.96):
    """Wilson 95% 신뢰구간"""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_stats(df, key_x, key_y):
    valid = df[~df["winner"].isin(["error"])]
    n = len(valid)
    total = len(df)
    wins_x = (valid["winner"] == key_x).sum()
    wins_y = (valid["winner"] == key_y).sum()
    ties = (valid["winner"] == "tie").sum()

    ci_x_lo, ci_x_hi = wilson_ci(wins_x, n)
    ci_y_lo, ci_y_hi = wilson_ci(wins_y, n)

    return {
        "n": n,
        "total": total,
        "wins_x": int(wins_x),
        "wins_y": int(wins_y),
        "ties": int(ties),
        "winrate_x": wins_x / n if n else 0,
        "winrate_y": wins_y / n if n else 0,
        "ci_x": (ci_x_lo, ci_x_hi),
        "ci_y": (ci_y_lo, ci_y_hi),
        "mean_score_x": valid["score_x"].mean(),
        "std_score_x": valid["score_x"].std(),
        "mean_score_y": valid["score_y"].mean(),
        "std_score_y": valid["score_y"].std(),
    }


def parse_results(all_results, key_x, key_y):
    rows = []
    for pd_ in all_results:
        for r in pd_["runs"]:
            j = r.get("judge", {})
            rows.append(
                {
                    "persona": pd_["persona"],
                    "run": r.get("run", 0),
                    "winner": j.get("winner", "error"),
                    "score_x": j.get("score_x", 0),
                    "score_y": j.get("score_y", 0),
                    "reason": j.get("reason", ""),
                }
            )
    return pd.DataFrame(rows)


# ==========================================
# 10. Streamlit UI
# ==========================================
def render_stat_cards(stats, key_x, key_y):
    name_x = MODEL_REGISTRY[key_x]["short"]
    name_y = MODEL_REGISTRY[key_y]["short"]
    color_x = MODEL_REGISTRY[key_x]["color"]
    color_y = MODEL_REGISTRY[key_y]["color"]
    n = stats["n"]

    st.markdown("#### 통계 요약")
    c1, c2, c3, c4 = st.columns(4)

    # 승률 + 신뢰구간
    wr_x = stats["winrate_x"] * 100
    wr_y = stats["winrate_y"] * 100
    ci_x = stats["ci_x"]
    ci_y = stats["ci_y"]

    c1.metric(
        f"{name_x} 승률",
        f"{wr_x:.1f}%",
        f"95% CI [{ci_x[0]*100:.1f}% – {ci_x[1]*100:.1f}%]",
    )
    c2.metric(
        f"{name_y} 승률",
        f"{wr_y:.1f}%",
        f"95% CI [{ci_y[0]*100:.1f}% – {ci_y[1]*100:.1f}%]",
    )
    c3.metric(
        f"{name_x} 평균 점수",
        f"{stats['mean_score_x']:.2f} / 5",
        f"σ = {stats['std_score_x']:.2f}",
    )
    c4.metric(
        f"{name_y} 평균 점수",
        f"{stats['mean_score_y']:.2f} / 5",
        f"σ = {stats['std_score_y']:.2f}",
    )

    total = stats.get("total", n)
    errors = total - n
    st.caption(
        f"유효 샘플: **{n}** / {total}  |  "
        f"무승부: **{stats['ties']}**  |  "
        f"오류: **{errors}**"
    )


def render_charts(df, stats, key_x, key_y):
    name_x = MODEL_REGISTRY[key_x]["short"]
    name_y = MODEL_REGISTRY[key_y]["short"]
    color_x = MODEL_REGISTRY[key_x]["color"]
    color_y = MODEL_REGISTRY[key_y]["color"]

    valid = df[~df["winner"].isin(["error"])].copy()

    if valid.empty:
        st.warning(
            "유효한 평가 결과가 없습니다. 오류 없이 완료된 실행이 있는지 확인하세요."
        )
        return

    # ── 차트 1: 전체 승패 파이 대체 (가로 단일 막대) ────────────────────
    st.subheader("전체 승패")
    win_df = pd.DataFrame(
        [
            {"결과": f"{name_x} 승리", "횟수": stats["wins_x"], "color": color_x},
            {"결과": f"{name_y} 승리", "횟수": stats["wins_y"], "color": color_y},
            {"결과": "무승부", "횟수": stats["ties"], "color": "#9ca3af"},
        ]
    )
    bar_total = (
        alt.Chart(win_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("횟수:Q", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("결과:N", sort=None),
            color=alt.Color(
                "결과:N",
                scale=alt.Scale(
                    domain=win_df["결과"].tolist(), range=win_df["color"].tolist()
                ),
                legend=None,
            ),
            tooltip=["결과", "횟수"],
        )
        .properties(height=130)
    )
    st.altair_chart(bar_total, use_container_width=True)

    # ── 차트 2: 페르소나별 승패 ────────────────────────────────────────
    st.subheader("페르소나별 승패")
    persona_win = valid.copy()
    persona_win["winner_label"] = (
        persona_win["winner"]
        .map(
            {
                key_x: f"{name_x} 승리",
                key_y: f"{name_y} 승리",
                "tie": "무승부",
            }
        )
        .fillna("기타")
    )

    pw = (
        persona_win.groupby(["persona", "winner_label"])
        .size()
        .reset_index(name="count")
    )
    chart_persona = (
        alt.Chart(pw)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="횟수"),
            y=alt.Y("persona:N", title="페르소나"),
            color=alt.Color(
                "winner_label:N",
                scale=alt.Scale(
                    domain=[f"{name_x} 승리", f"{name_y} 승리", "무승부"],
                    range=[color_x, color_y, "#9ca3af"],
                ),
            ),
            tooltip=["persona", "winner_label", "count"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart_persona, use_container_width=True)

    # ── 차트 3: 페르소나별 평균 점수 ± 표준편차 ─────────────────────────
    st.subheader("페르소나별 평균 점수 (± 표준편차)")
    score_rows = []
    for persona, grp in valid.groupby("persona"):
        score_rows.append(
            {
                "페르소나": persona,
                "모델": name_x,
                "평균": grp["score_x"].mean(),
                "std": grp["score_x"].std(),
            }
        )
        score_rows.append(
            {
                "페르소나": persona,
                "모델": name_y,
                "평균": grp["score_y"].mean(),
                "std": grp["score_y"].std(),
            }
        )

    if not score_rows:
        st.info("점수 데이터가 없습니다.")
    else:
        score_df = pd.DataFrame(score_rows)
        score_df["std"] = score_df["std"].fillna(0)
        score_df["상한"] = score_df["평균"] + score_df["std"]
        score_df["하한"] = (score_df["평균"] - score_df["std"]).clip(lower=0)

        base = alt.Chart(score_df)
        bars = base.mark_bar(opacity=0.85).encode(
            x=alt.X("평균:Q", scale=alt.Scale(domain=[0, 5]), title="평균 점수"),
            y=alt.Y("페르소나:N"),
            color=alt.Color(
                "모델:N",
                scale=alt.Scale(domain=[name_x, name_y], range=[color_x, color_y]),
            ),
            xOffset="모델:N",
            tooltip=[
                "페르소나",
                "모델",
                alt.Tooltip("평균:Q", format=".2f"),
                alt.Tooltip("std:Q", format=".2f", title="σ"),
            ],
        )
        error = base.mark_errorbar(extent="stdev").encode(
            x=alt.X("평균:Q"),
            xError=alt.XError("std:Q"),
            y=alt.Y("페르소나:N"),
            yOffset="모델:N",
            color=alt.Color(
                "모델:N",
                scale=alt.Scale(domain=[name_x, name_y], range=[color_x, color_y]),
            ),
        )
        st.altair_chart((bars + error).properties(height=250), use_container_width=True)

    # ── 판정 이유 테이블 ───────────────────────────────────────────────
    st.subheader("판정 이유 샘플")
    sample = (
        valid[
            [
                "persona",
                "run",
                "winner_label" if "winner_label" in valid.columns else "winner",
                "score_x",
                "score_y",
                "reason",
            ]
        ]
        .rename(columns={"score_x": f"{name_x} 점수", "score_y": f"{name_y} 점수"})
        .head(25)
    )
    st.dataframe(sample, use_container_width=True)


# ==========================================
# 11. 진실성 검증 탭 (Human-in-the-Loop + AI 혜택 정확도)
# ==========================================
def render_verification_tab(resources: dict):
    st.header("진실성 검증")
    st.caption(
        "카드 이름 존재 여부 확인 · AI(gpt-5-mini) 혜택 정확도 검증 · 사람 판정 수정"
    )

    # ── 파일 선택 (OUTPUT_DIR + EVAL_FINAL_DIRS 통합 스캔) ────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_files = _scan_eval_files(include_corrected=False)

    if not source_files:
        st.warning(
            f"`{OUTPUT_DIR}/` 폴더에 평가 파일이 없습니다. 먼저 평가를 실행하세요."
        )
        return

    selected_file = st.selectbox(
        "검증할 평가 파일 선택",
        source_files,
        format_func=lambda x: os.path.basename(x),
    )

    with open(selected_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "key_x" not in data or "key_y" not in data:
        st.error(
            f"`{os.path.basename(selected_file)}`은 구버전 포맷입니다. "
            f"`key_x` / `key_y` 필드가 없습니다. "
            f"최신 버전으로 평가를 재실행하거나 다른 파일을 선택하세요."
        )
        return

    key_x = data["key_x"]
    key_y = data["key_y"]

    if key_x not in MODEL_REGISTRY or key_y not in MODEL_REGISTRY:
        st.error(f"파일의 모델 키 (`{key_x}`, `{key_y}`)가 MODEL_REGISTRY에 없습니다.")
        return

    name_x = MODEL_REGISTRY[key_x]["short"]
    name_y = MODEL_REGISTRY[key_y]["short"]
    color_x = MODEL_REGISTRY[key_x]["color"]
    color_y = MODEL_REGISTRY[key_y]["color"]

    # ── cards.json 로드 ────────────────────────────────────────────────────
    with open("data/cards.json", "r", encoding="utf-8") as f:
        cards_json = json.load(f)
    real_card_names = {c["card_name"] for c in cards_json}

    # 혜택 정확도 검증용: card_name → 혜택 요약 텍스트
    cards_lookup: dict[str, str] = {}
    for c in cards_json:
        benefit_lines = [
            f"- [{b['benefits_name']}] {b['benefits_summary']}"
            for b in c.get("benefits_structured", [])
            if b.get("benefits_name") != "유의사항"
        ]
        cards_lookup[c["card_name"]] = "\n".join(benefit_lines)

    # 페르소나 이름 → 소비 패턴 전문 텍스트 (AI 검증 컨텍스트용)
    pattern_lookup: dict[str, str] = {tc["name"]: tc["pattern"] for tc in TEST_CASES}

    # ── 세션 상태 초기화 ───────────────────────────────────────────────────
    correction_key = f"corrections_{os.path.basename(selected_file)}"
    if correction_key not in st.session_state:
        st.session_state[correction_key] = {}

    corrections = st.session_state[correction_key]

    # ── 상단: 진행 현황 요약 ───────────────────────────────────────────────
    total_runs = sum(len(p["runs"]) for p in data["results"])
    corrected_count = len(corrections)
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 Run", total_runs)
    c2.metric("수정 완료", corrected_count)
    c3.metric("미검증", total_runs - corrected_count)

    st.divider()

    # ── 페르소나별 검증 패널 ──────────────────────────────────────────────
    winner_options = [key_x, key_y, "tie"]
    winner_labels = [f"{name_x} 승리", f"{name_y} 승리", "무승부"]

    for persona_data in data["results"]:
        persona = persona_data["persona"]
        st.subheader(f"📌 {persona}")

        for run_data in persona_data["runs"]:
            run_num = run_data.get("run", 0)
            run_key = f"{persona}__run{run_num}"
            cards_x = run_data.get("cards_x", [])
            cards_y = run_data.get("cards_y", [])
            judge = corrections.get(run_key, run_data.get("judge", {}))
            is_error = run_data.get("judge", {}).get("winner") == "error"
            tag = (
                "✏️ 수정됨" if run_key in corrections else ("⚠️ 오류" if is_error else "")
            )

            with st.expander(
                f"Run {run_num} — 판정: **{winner_labels[winner_options.index(judge.get('winner', 'tie'))] if judge.get('winner') in winner_options else judge.get('winner','N/A')}**  {tag}"
            ):
                col_x, col_y = st.columns(2)

                # ── 1단계: 원본 데이터 검증 ───────────────────────────────
                with col_x:
                    st.markdown(f"**{name_x} 추천 카드**")
                    if not cards_x:
                        st.caption("카드 정보 없음")
                    for card in cards_x:
                        exists = card in real_card_names
                        icon = "✅" if exists else "❌"
                        st.markdown(f"{icon} `{card}`")
                        gorilla_url = f"https://www.card-gorilla.com/search/all?keyword={quote(card)}"
                        st.link_button(
                            "카드고릴라 검색 →", gorilla_url, use_container_width=False
                        )

                with col_y:
                    st.markdown(f"**{name_y} 추천 카드**")
                    if not cards_y:
                        st.caption("카드 정보 없음")
                    for card in cards_y:
                        exists = card in real_card_names
                        icon = "✅" if exists else "❌"
                        st.markdown(f"{icon} `{card}`")
                        gorilla_url = f"https://www.card-gorilla.com/search/all?keyword={quote(card)}"
                        st.link_button(
                            "카드고릴라 검색 →", gorilla_url, use_container_width=False
                        )

                # ── 2단계: AI 혜택 정확도 검증 ───────────────────────────
                st.divider()
                st.markdown("**AI 혜택 정확도 검증** (gpt-5-mini)")
                ai_key = f"ai_verify_{run_key}"
                resp_x = run_data.get("resp_x", "")
                resp_y = run_data.get("resp_y", "")
                user_pattern_text = pattern_lookup.get(persona, persona)

                if st.button(
                    "🤖 혜택 정확도 자동 검증",
                    key=f"ai_btn_{run_key}",
                    disabled=not (resp_x or resp_y),
                ):
                    with st.spinner("검증 중..."):
                        vx = verify_benefits_accuracy(
                            resources["openai_judge"],
                            user_pattern_text,
                            cards_x,
                            cards_lookup,
                            resp_x,
                            name_x,
                        )
                        vy = verify_benefits_accuracy(
                            resources["openai_judge"],
                            user_pattern_text,
                            cards_y,
                            cards_lookup,
                            resp_y,
                            name_y,
                        )
                    st.session_state[ai_key] = {"x": vx, "y": vy}

                if ai_key in st.session_state:
                    vr = st.session_state[ai_key]
                    vcol_x, vcol_y = st.columns(2)
                    for vcol, side, vres in [
                        (vcol_x, name_x, vr["x"]),
                        (vcol_y, name_y, vr["y"]),
                    ]:
                        with vcol:
                            verdict = vres.get("verdict", "?")
                            score = vres.get("accuracy_score", 0)
                            hallu = vres.get("hallucination_detected", False)
                            icon = (
                                "✅"
                                if verdict == "PASS"
                                else ("⚠️" if verdict == "SKIP" else "❌")
                            )
                            st.markdown(
                                f"{icon} **{side}** — 정확도 `{score}/5`  |  환각: `{'있음' if hallu else '없음'}`"
                            )
                            issues = vres.get("issues", "")
                            if issues and issues != "없음":
                                st.caption(f"문제점: {issues}")

                # ── 3단계: 판정 수정 ──────────────────────────────────────
                st.divider()
                st.markdown("**판정 수정**")

                current_winner = judge.get("winner", "tie")
                winner_idx = (
                    winner_options.index(current_winner)
                    if current_winner in winner_options
                    else 2
                )

                new_winner = st.radio(
                    "승자 선택",
                    options=winner_options,
                    format_func=lambda x: winner_labels[winner_options.index(x)],
                    index=winner_idx,
                    horizontal=True,
                    key=f"winner_{run_key}",
                )

                sc1, sc2 = st.columns(2)
                new_score_x = sc1.slider(
                    f"{name_x} 점수",
                    1,
                    5,
                    int(judge.get("score_x", 3)),
                    key=f"sx_{run_key}",
                )
                new_score_y = sc2.slider(
                    f"{name_y} 점수",
                    1,
                    5,
                    int(judge.get("score_y", 3)),
                    key=f"sy_{run_key}",
                )
                new_reason = st.text_input(
                    "판정 이유",
                    value=judge.get("reason", ""),
                    key=f"reason_{run_key}",
                )

                if st.button("💾 이 Run 수정 저장", key=f"save_{run_key}"):
                    corrections[run_key] = {
                        "winner": new_winner,
                        "score_x": new_score_x,
                        "score_y": new_score_y,
                        "reason": new_reason,
                        "human_corrected": True,
                    }
                    st.session_state[correction_key] = corrections
                    st.success(f"Run {run_num} 수정 저장 완료")
                    st.rerun()

        st.write("")

    # ── 수정본 파일 생성 + 통계 ───────────────────────────────────────────
    st.divider()
    col_save, col_stat = st.columns(2)

    corrected_file = selected_file.replace(".json", "_corrected.json")

    if col_save.button("📁 수정본 파일 생성", type="primary", use_container_width=True):
        import copy

        corrected_data = copy.deepcopy(data)
        for persona_data in corrected_data["results"]:
            persona = persona_data["persona"]
            for run_data in persona_data["runs"]:
                rk = f"{persona}__run{run_data.get('run', 0)}"
                if rk in corrections:
                    run_data["judge"] = corrections[rk]
        with open(corrected_file, "w", encoding="utf-8") as f:
            json.dump(corrected_data, f, ensure_ascii=False, indent=2)
        st.success(f"저장 완료: `{corrected_file}`")

    if col_stat.button("📊 수정본 통계 보기", use_container_width=True):
        if os.path.exists(corrected_file):
            with open(corrected_file, "r", encoding="utf-8") as f:
                corrected = json.load(f)
            kx, ky = corrected["key_x"], corrected["key_y"]
            df_c = parse_results(corrected["results"], kx, ky)
            stats_c = compute_stats(df_c, kx, ky)
            st.markdown("### 수정본 기반 통계")
            render_stat_cards(stats_c, kx, ky)
            st.divider()
            render_charts(df_c, stats_c, kx, ky)
        else:
            st.warning("수정본 파일이 없습니다. 먼저 '수정본 파일 생성'을 눌러주세요.")


# ==========================================
# 12. 데이터 병합 및 통계 탭
# ==========================================
def render_merge_stats_tab():
    st.header("데이터 병합 및 통계")
    st.caption("여러 평가 파일을 선택해 병합하고 발표용 통계·그래프를 확인합니다.")

    dir_labels = {os.path.basename(d): d for d in EVAL_FINAL_DIRS}

    # ── 디렉토리별 파일 선택 ───────────────────────────────────────────────
    selected_per_dir: dict[str, list[str]] = {}
    for label, dirpath in dir_labels.items():
        files = sorted(glob.glob(f"{dirpath}/*.json")) if os.path.isdir(dirpath) else []
        files = [f for f in files if "_corrected" not in f]
        if not files:
            continue
        with st.expander(f"📂 {label}  ({len(files)}개 파일)", expanded=True):
            chosen = st.multiselect(
                "병합할 파일",
                files,
                default=files,
                format_func=os.path.basename,
                key=f"merge_sel_{label}",
            )
            if chosen:
                selected_per_dir[label] = chosen

    if not selected_per_dir:
        st.info("평가 파일이 없습니다. 디렉토리를 확인하세요.")
        return

    if st.button("📊 병합 및 통계 계산", type="primary"):
        merged_data: dict[str, dict] = {}
        for label, file_list in selected_per_dir.items():
            all_results: list = []
            kx, ky = None, None
            for fp in file_list:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if kx is None:
                    kx = raw.get("key_x")
                    ky = raw.get("key_y")
                all_results.extend(raw.get("results", []))
            if not kx or not ky or not all_results:
                continue
            df = parse_results(all_results, kx, ky)
            stats = compute_stats(df, kx, ky)
            merged_data[label] = {
                "df": df,
                "kx": kx,
                "ky": ky,
                "stats": stats,
                "file_count": len(file_list),
                "total_runs": len(df),
            }
        st.session_state["merge_data"] = merged_data

    if "merge_data" not in st.session_state or not st.session_state["merge_data"]:
        return

    merged_data = st.session_state["merge_data"]

    # ── 통계 요약 카드 ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("통계 요약")
    stat_cols = st.columns(len(merged_data))
    for col, (label, d) in zip(stat_cols, merged_data.items()):
        with col:
            st.markdown(f"**{label}** ({d['file_count']}개 파일 / {d['total_runs']}회)")
            render_stat_cards(d["stats"], d["kx"], d["ky"])

    # ── 데이터셋 간 승률 비교 막대 차트 ──────────────────────────────────
    if len(merged_data) > 1:
        st.divider()
        st.subheader("데이터셋 간 승률 비교")
        rows = []
        for label, d in merged_data.items():
            s = d["stats"]
            nx = MODEL_REGISTRY.get(d["kx"], {}).get("short", d["kx"])
            ny = MODEL_REGISTRY.get(d["ky"], {}).get("short", d["ky"])
            rows.extend(
                [
                    {
                        "데이터셋": label,
                        "모델": nx,
                        "승률(%)": round(s["winrate_x"] * 100, 1),
                    },
                    {
                        "데이터셋": label,
                        "모델": ny,
                        "승률(%)": round(s["winrate_y"] * 100, 1),
                    },
                    {
                        "데이터셋": label,
                        "모델": "무승부",
                        "승률(%)": round(s["ties"] / s["n"] * 100, 1) if s["n"] else 0,
                    },
                ]
            )
        cmp_df = pd.DataFrame(rows)
        cmp_chart = (
            alt.Chart(cmp_df)
            .mark_bar()
            .encode(
                x=alt.X("데이터셋:N"),
                y=alt.Y("승률(%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("모델:N"),
                xOffset="모델:N",
                tooltip=["데이터셋", "모델", "승률(%)"],
            )
            .properties(height=300)
        )
        st.altair_chart(cmp_chart, use_container_width=True)

        # 평균 점수 비교
        st.subheader("데이터셋 간 평균 점수 비교")
        score_rows = []
        for label, d in merged_data.items():
            s = d["stats"]
            nx = MODEL_REGISTRY.get(d["kx"], {}).get("short", d["kx"])
            ny = MODEL_REGISTRY.get(d["ky"], {}).get("short", d["ky"])
            score_rows.extend(
                [
                    {
                        "데이터셋": label,
                        "모델": nx,
                        "평균점수": round(s["mean_score_x"], 2),
                        "std": round(s["std_score_x"], 2),
                    },
                    {
                        "데이터셋": label,
                        "모델": ny,
                        "평균점수": round(s["mean_score_y"], 2),
                        "std": round(s["std_score_y"], 2),
                    },
                ]
            )
        sc_df = pd.DataFrame(score_rows)
        sc_chart = (
            alt.Chart(sc_df)
            .mark_bar()
            .encode(
                x=alt.X("데이터셋:N"),
                y=alt.Y("평균점수:Q", scale=alt.Scale(domain=[0, 5])),
                color=alt.Color("모델:N"),
                xOffset="모델:N",
                tooltip=[
                    "데이터셋",
                    "모델",
                    alt.Tooltip("평균점수:Q", format=".2f"),
                    alt.Tooltip("std:Q", format=".2f", title="σ"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(sc_chart, use_container_width=True)

    # ── 데이터셋별 상세 차트 ──────────────────────────────────────────────
    st.divider()
    st.subheader("데이터셋별 상세 차트")
    for label, d in merged_data.items():
        with st.expander(f"📈 {label} 상세", expanded=False):
            render_charts(d["df"], d["stats"], d["kx"], d["ky"])

    # ── JSON 내보내기 ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("통계 데이터 내보내기")

    export_payload: dict = {}
    for label, d in merged_data.items():
        s = d["stats"]
        nx = MODEL_REGISTRY.get(d["kx"], {}).get("short", d["kx"])
        ny = MODEL_REGISTRY.get(d["ky"], {}).get("short", d["ky"])

        # 페르소나별 세부 통계
        persona_stats = []
        for persona, grp in d["df"].groupby("persona"):
            valid_grp = grp[~grp["winner"].isin(["error"])]
            n_p = len(valid_grp)
            persona_stats.append(
                {
                    "persona": persona,
                    "total_runs": n_p,
                    f"{nx}_wins": int((valid_grp["winner"] == d["kx"]).sum()),
                    f"{ny}_wins": int((valid_grp["winner"] == d["ky"]).sum()),
                    "ties": int((valid_grp["winner"] == "tie").sum()),
                    f"{nx}_avg_score": (
                        round(valid_grp["score_x"].mean(), 2) if n_p else 0
                    ),
                    f"{ny}_avg_score": (
                        round(valid_grp["score_y"].mean(), 2) if n_p else 0
                    ),
                }
            )

        export_payload[label] = {
            "dataset": label,
            "model_x": {"key": d["kx"], "short": nx},
            "model_y": {"key": d["ky"], "short": ny},
            "file_count": d["file_count"],
            "summary": {
                "total_runs": s["total"],
                "valid_runs": s["n"],
                "errors": s["total"] - s["n"],
                f"{nx}_wins": s["wins_x"],
                f"{ny}_wins": s["wins_y"],
                "ties": s["ties"],
                f"{nx}_winrate_pct": round(s["winrate_x"] * 100, 1),
                f"{ny}_winrate_pct": round(s["winrate_y"] * 100, 1),
                f"{nx}_winrate_ci_95": [
                    round(s["ci_x"][0] * 100, 1),
                    round(s["ci_x"][1] * 100, 1),
                ],
                f"{ny}_winrate_ci_95": [
                    round(s["ci_y"][0] * 100, 1),
                    round(s["ci_y"][1] * 100, 1),
                ],
                f"{nx}_avg_score": round(s["mean_score_x"], 2),
                f"{nx}_std_score": round(s["std_score_x"], 2),
                f"{ny}_avg_score": round(s["mean_score_y"], 2),
                f"{ny}_std_score": round(s["std_score_y"], 2),
            },
            "persona_details": persona_stats,
        }

    export_json = json.dumps(export_payload, ensure_ascii=False, indent=2)

    st.download_button(
        label="⬇️ 병합 통계 JSON 다운로드",
        data=export_json.encode("utf-8"),
        file_name="merged_stats.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )

    with st.expander("미리보기"):
        st.json(export_payload)


def main():
    st.set_page_config(page_title="모델 비교 평가", layout="wide")
    st.title("카드 추천 모델 비교 평가")

    resources = init_resources()

    # ── 사이드바: 모델 선택 ──────────────────────────────────────────────
    with st.sidebar:
        st.header("모델 선택")
        model_options = {v["name"]: k for k, v in MODEL_REGISTRY.items()}
        names = list(model_options.keys())

        sel_x_name = st.selectbox("비교 모델 X (왼쪽)", names, index=0)
        sel_y_name = st.selectbox("비교 모델 Y (오른쪽)", names, index=1)
        key_x = model_options[sel_x_name]
        key_y = model_options[sel_y_name]

        if key_x == key_y:
            st.warning("서로 다른 모델을 선택하세요.")

        st.divider()
        for info in MODEL_REGISTRY.values():
            st.markdown(f"**{info['name']}**")
            st.caption(info["description"])
            st.write("")

    # ── 메인: 버튼 ──────────────────────────────────────────────────────
    name_x = MODEL_REGISTRY[key_x]["name"]
    name_y = MODEL_REGISTRY[key_y]["name"]
    st.caption(
        f"비교: **{name_x}** vs **{name_y}**  |  "
        f"케이스 {len(TEST_CASES)}개 × {NUM_RUNS}회 = 총 {len(TEST_CASES)*NUM_RUNS}판  |  Judge: `{JUDGE_MODEL}`"
    )

    tab_eval, tab_verify, tab_merge = st.tabs(
        ["📊 평가 실행", "🔍 진실성 검증", "📈 병합 통계"]
    )

    with tab_eval:
        c1, c2 = st.columns([1, 3])
        run_btn = c1.button(
            "평가 시작",
            type="primary",
            use_container_width=True,
            disabled=(key_x == key_y),
        )
        load_btn = c2.button("기존 결과 불러오기", use_container_width=True)
        st.divider()

        output_file = f"{OUTPUT_DIR}/{key_x}_vs_{key_y}.json"

        if run_btn:
            all_results, kx, ky = run_evaluation(key_x, key_y, resources)
            st.session_state["eval_df"] = parse_results(all_results, kx, ky)
            st.session_state["eval_kx"] = kx
            st.session_state["eval_ky"] = ky
            st.session_state["eval_raw"] = all_results

        if load_btn:
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                kx, ky = saved["key_x"], saved["key_y"]
                st.session_state["eval_df"] = parse_results(saved["results"], kx, ky)
                st.session_state["eval_kx"] = kx
                st.session_state["eval_ky"] = ky
                st.session_state["eval_raw"] = saved["results"]
                st.success(f"`{output_file}` 불러오기 완료")
            else:
                st.warning(f"`{output_file}` 파일이 없습니다. 먼저 평가를 실행하세요.")

        if "eval_df" in st.session_state:
            df = st.session_state["eval_df"]
            kx = st.session_state["eval_kx"]
            ky = st.session_state["eval_ky"]
            stats = compute_stats(df, kx, ky)

            render_stat_cards(stats, kx, ky)
            st.divider()
            render_charts(df, stats, kx, ky)

            with st.expander("원본 결과 JSON"):
                st.json(st.session_state["eval_raw"])

    with tab_verify:
        render_verification_tab(resources)

    with tab_merge:
        render_merge_stats_tab()


if __name__ == "__main__":
    main()
