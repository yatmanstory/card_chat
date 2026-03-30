import os
import re
import json
import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from streamlit_agraph import agraph, Node, Edge, Config


# Pre_Filter_Model_app.py  = Model-B, PreFiltering + RRF 가중치, 멀티 쿼리 적용

# ==========================================
# 1. 초기 설정 및 환경 변수
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_DIR = "./VectorStores_Card"

embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
# 카드 단위 집계를 위한 raw chromadb 클라이언트
_chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
raw_collection = _chroma_client.get_collection(
    _chroma_client.list_collections()[0].name
)

# ── 카테고리 데이터 전역 로드 ─────────────────────────────────────────────
with open("data/categories_rows.json", "r", encoding="utf-8") as f:
    CATEGORIES = json.load(f)

CATEGORY_MAP = {c["name"]: json.loads(c["mapped_names"]) for c in CATEGORIES}


# ==========================================
# 2. 세션 상태 (Session State) 초기화
# ==========================================
# 기술 부채 :
# hitory 세션에 등록되고 실질적으로 반환되지 않아 과거 대화 내역 삭제, 이번 프로젝트에서는 구현 X 시간 없음
# 따라서 멀티턴 구현도 X
# 스트리밍 구현 X 시간 없음, 추후 과제로 남기자
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! Card Concierge입니다. 고객님의 소비 패턴(예: 배달, 교통, 쇼핑 등)이나 궁금한 점을 편하게 말씀해 주시면, 최적의 카드를 추천해 드립니다.",
                "cards": [],
            }
        ]
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = []
    if "card_index" not in st.session_state:
        st.session_state.card_index = 0
    if "last_clicked_id" not in st.session_state:
        st.session_state.last_clicked_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()


# ==========================================
# 3. AI 및 DB 로직 (RAG, LLM 함수)
# ==========================================
def extract_consumption_pattern(user_text):
    category_list = "\n".join(f'- id={c["id"]}, name="{c["name"]}"' for c in CATEGORIES)
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """**사용자의 소비 패턴을 분석**하여 JSON 형식으로 응답하세요.
반드시 'categories'와 'search_profile' 두 가지 키를 포함해야 합니다.

1. 'categories': 사용자 질의에서 가장 지출 비중이 높은 핵심 분야만 **최대 4개** 선정하여 [id, name, percent] 리스트로 담으세요. 4개를 초과하면 안 됩니다.
2. 선정한 4개 항목의 percent 합계는 반드시 100이 되어야 합니다.
3. 'search_profile': 사용자의 상황과 필요한 혜택을 모든 키워드를 포함해 상세한 문장으로 재작성하세요.
   (예: "20대 대학생, 월 80만원 지출, 배달앱, 카페, 대중교통, OTT 구독 할인 혜택이 필요한 상황")

[허용 카테고리 목록]:
{category_list}

응답 예시:
{{
    "categories": [
        {{"id": 3, "name": "카페/디저트", "percent": 30}},
        {{"id": 4, "name": "배달앱", "percent": 40}}
    ],
    "search_profile": "20대 대학생으로 배달음식과 카페 이용 비중이 높으며 교통비와 OTT 구독 할인이 필요한 상황"
}}
""",
            ),
            ("human", "{question}"),
        ]
    )

    chain = extract_prompt | chat_model
    response = chain.invoke({"question": user_text, "category_list": category_list})
    clean_json = response.content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(clean_json)
        if "categories" in data:
            data["categories"] = sorted(
                data["categories"], key=lambda x: x.get("percent", 0), reverse=True
            )[:4]
        return data
    except (json.JSONDecodeError, KeyError):
        return {"categories": [], "search_profile": user_text}


def get_search_keywords(categories):
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


def _get_card_match_count(keywords):
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


def format_fee(fee):
    if not fee or fee == "-":
        return "정보 없음"
    if fee == "없음":
        return "무료"
    try:
        return f"{int(fee):,}원"
    except:
        return fee + "원"


def _docs_to_cards(docs):
    seen = set()
    cards = []
    for doc in docs:
        meta = doc.metadata
        card_id = meta.get("card_id", "")
        if card_id in seen:
            continue
        seen.add(card_id)

        try:
            benefits = json.loads(meta.get("benefits_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            benefits = []

        cards.append(
            {
                "card_id": meta.get("card_id", ""),
                "card_name": meta.get("card_name", "-"),
                "card_company": meta.get("card_company", "-"),
                "image_url": meta.get("image_url", ""),
                "detail_url": meta.get("detail_url", "#"),
                "fee": meta.get("annual_fee_domestic", "-"),
                "condition": meta.get("base_performance", "-"),
                "benefits": benefits,
                "badge": "RECOMMENDED",
                "badge_color": "#3b82f6",
                "btn_text": "자세히 보기",
            }
        )
    return cards


def search_similar_cards(analysis_result, top_k=3, card_pool=9):
    """
    [app2.py - 멀티쿼리 + RRF 스코어링 버전]
    benefit_keywords 필터 → card_id 후보 풀 결정
    → 카테고리별 멀티쿼리 + 통합 쿼리로 RRF 스코어링
    → 스코어 상위 top_k 카드 선별
    """
    categories = analysis_result.get("categories", [])
    search_profile = analysis_result.get("search_profile", "")

    try:
        # 1. 후보 card_id 풀 결정: 1·2순위 키워드 교집합 우선 → 합집합 폴백
        top1_keywords = get_search_keywords(categories[:1])
        top2_keywords = (
            get_search_keywords(categories[1:2]) if len(categories) > 1 else []
        )
        all_keywords = get_search_keywords(categories)
        id_filter = None
        candidate_mode = "none"

        if top1_keywords and top2_keywords:
            top1_count = _get_card_match_count(top1_keywords)
            top2_count = _get_card_match_count(top2_keywords)
            intersection_ids = set(top1_count) & set(top2_count)

            if len(intersection_ids) >= top_k:
                # 교집합: 1·2순위 혜택을 모두 가진 카드 → 전체 키워드 매칭 횟수로 랭킹
                all_count = _get_card_match_count(all_keywords)
                top_card_ids = sorted(
                    intersection_ids,
                    key=lambda x: all_count.get(x, 0),
                    reverse=True,
                )[:card_pool]
                candidate_mode = "intersection"
            else:
                # 폴백: 전체 키워드 합집합
                all_count = _get_card_match_count(all_keywords)
                top_card_ids = sorted(
                    all_count, key=lambda x: all_count[x], reverse=True
                )[:card_pool]
                candidate_mode = "union"
        elif all_keywords:
            all_count = _get_card_match_count(all_keywords)
            top_card_ids = sorted(all_count, key=lambda x: all_count[x], reverse=True)[
                :card_pool
            ]
            candidate_mode = "union"
        else:
            top_card_ids = []

        if len(top_card_ids) >= top_k:
            id_filter = (
                {"card_id": {"$eq": top_card_ids[0]}}
                if len(top_card_ids) == 1
                else {"$or": [{"card_id": {"$eq": cid}} for cid in top_card_ids]}
            )

        # 2. 멀티쿼리 구성: 카테고리별 비중 강조 쿼리 + 통합 강조 쿼리
        rank_labels = ["1순위", "2순위", "3순위", "4순위"]
        queries: list[tuple[str, float]] = [
            (
                f"[{rank_labels[i] if i < 4 else f'{i+1}순위'} {cat['name']} {cat['percent']}%] "
                f"{cat['name']} 집중 할인 혜택 카드",
                cat["percent"],
            )
            for i, cat in enumerate(categories)
        ]
        top_cats = categories[:2]
        top_emphasis = ", ".join(
            f"{c['name']}({c['percent']}%) 집중 할인" for c in top_cats
        )
        top_percent = max((c["percent"] for c in categories), default=50)
        integrated_query = (
            f"{top_emphasis}. {search_profile}. {top_cats[0]['name']} 할인이 가장 중요."
            if top_cats
            else search_profile
        )
        queries.append((integrated_query, top_percent))

        # 3. card_id별 RRF 스코어 집계 (score = percent / (rank + 1))
        MAX_CHUNKS_PER_CARD = 10  # LLM 컨텍스트 토큰 초과 방지
        card_score: dict[str, float] = {}
        card_docs: dict[str, list] = {}
        card_seen_contents: dict[str, set] = {}  # 카드별 중복 청크 방지

        for query_text, weight in queries:
            search_k = 9 if id_filter else 15
            docs = vectordb.similarity_search(
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

        # 4. 스코어 상위 top_k card_id 선별
        top_card_ids = sorted(card_score, key=lambda x: card_score[x], reverse=True)[
            :top_k
        ]

        # 5. 해당 카드 docs로 카드 정보 및 컨텍스트 구성
        filtered_docs = [
            doc
            for cid in top_card_ids
            for doc in card_docs.get(cid, [])
            if doc.metadata.get("card_id") == cid
        ]
        cards = _docs_to_cards(filtered_docs)[:top_k]

        return cards, filtered_docs, candidate_mode

    except Exception as e:
        st.error(f"카드 검색 중 오류: {e}")
        return [], [], "error"


def search_similar_cards_by_category(category, top_k=3):
    try:
        cat_name = category.get("name", "")
        keywords = CATEGORY_MAP.get(cat_name, [cat_name])
        chroma_filter = _build_keyword_filter(keywords)

        docs = []
        if chroma_filter:
            docs = vectordb.similarity_search(
                cat_name, k=top_k * 2, filter=chroma_filter
            )

        if not docs:
            docs = vectordb.similarity_search(cat_name, k=top_k * 2)

        return _docs_to_cards(docs)[:top_k]

    except Exception as e:
        st.error(f"카테고리 검색 중 오류: {e}")
        return []


def generate_chat_response(user_text, docs, categories):
    priority_text = " > ".join(f"{c['name']}({c['percent']}%)" for c in categories)

    card_contents: dict[str, list[str]] = {}
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        cleaned = re.sub(
            r"\[유의사항\].*?(?=\n\[|\Z)", "", doc.page_content, flags=re.DOTALL
        ).strip()
        if cleaned:
            card_contents.setdefault(name, []).append(cleaned)

    context = "\n\n---\n\n".join(
        f"【 카드명: {card_name} 】\n" + "\n\n".join(chunks)
        for card_name, chunks in card_contents.items()
    )

    formatted_history = ""
    for msg in st.session_state.chat_history.messages:
        role = "사용자" if msg.type == "human" else "AI"
        formatted_history += f"{role}: {msg.content}\n"

    card_names = list(card_contents.keys())
    card_name_1 = card_names[0] if len(card_names) > 0 else "카드 1"
    card_name_2 = card_names[1] if len(card_names) > 1 else "카드 2"
    card_name_3 = card_names[2] if len(card_names) > 2 else "카드 3"

    prompt = f"""당신은 카드 추천 전문가입니다.
사용자의 소비 비중 우선순위: {priority_text}
비중이 높은 항목일수록 해당 혜택이 강한 카드를 우선 추천하세요.
아래 [추천 카드 목록]은 사용자에게 이미 화면에 보여주는 카드들입니다.

[추천 카드 목록]
{context}

[사용자 질문]
{user_text}

[답변 규칙 - 매우 중요]
1. 반드시 아래 카드 3개를 모두 순서대로 추천하세요. 어떠한 경우에도 생략하지 마세요.
   - {card_name_1}
   - {card_name_2}
   - {card_name_3}
2. 각 카드별로 소비 비중 상위 항목과 직접 연관된 혜택을 우선 선정하세요. ("모든가맹점" 같은 범용 카테고리보다 사용자가 명시한 항목 우선)
3. 카드명, 혜택명1, 혜택명2, 추천 이유는 반드시 각각 새로운 줄에서 시작하세요. 절대로 한 줄에 이어서 쓰지 마세요.
4. [금액 바인딩 필수] 사용자가 언급한 구체적 금액(예: "교통비 월 7만원")과 카드의 실제 혜택 조건(예: "대중교통 월 최대 5,000원 할인, 전월실적 30만원 이상")을 반드시 명시적으로 연결하세요.
   좋은 예: "월 15만원 배달 지출 기준, 이 카드의 배달앱 10% 할인 적용 시 월 최대 1만 5천원 절약 가능합니다."
   나쁜 예: "배달앱 할인 혜택이 있어 배달을 자주 이용하는 분께 적합합니다." (금액 연결 없는 일반적 설명 금지)
5. 문서에 없는 수치나 혜택은 절대 지어내지 마세요.
6. 사용자가 언급하지 않은 취미(예: 게임, 골프, 여행 등)를 사용자가 즐길 것이라고 추측하여 설명하지 마세요.
7. 추천하는 카드가 사용자가 요청한 영역의 혜택이 적다면, 있는 그대로의 혜택을 설명하되 거짓으로 필요성을 지어내지 마세요.

-- 출력 형식 (마크다운 문법을 그대로 사용하여 아래 형식을 정확히 따르세요) --
#### [**{card_name_1}**]
**혜택명1**: 혜택 요약 1줄 (실제 조건 포함)
**혜택명2**: 혜택 요약 1줄 (실제 조건 포함)

- 추천 이유: 사용자 금액 ↔ 카드 조건 연결 2~3문장

#### [**{card_name_2}**]
**혜택명1**: 혜택 요약 1줄 (실제 조건 포함)
**혜택명2**: 혜택 요약 1줄 (실제 조건 포함)

- 추천 이유: 사용자 금액 ↔ 카드 조건 연결 2~3문장

#### [**{card_name_3}**]
**혜택명1**: 혜택 요약 1줄 (실제 조건 포함)
**혜택명2**: 혜택 요약 1줄 (실제 조건 포함)

- 추천 이유: 사용자 금액 ↔ 카드 조건 연결 2~3문장
"""

    response = chat_model.invoke(prompt)
    reply = response.content

    st.session_state.chat_history.add_user_message(user_text)
    st.session_state.chat_history.add_ai_message(reply)

    return reply, prompt


# ==========================================
# 4. UI 컴포넌트 함수
# ==========================================
def render_3_column_cards(cards):
    if not cards:
        return

    cols = st.columns(3)
    for i, card in enumerate(cards[:3]):
        with cols[i]:
            with st.container(border=True):
                bg_style = (
                    f"background-image: url('{card.get('image_url')}'); background-size: cover; background-position: center;"
                    if card.get("image_url")
                    else "background: #ccc;"
                )

                st.markdown(
                    f"""
                    <div style="{bg_style} height: 200px; border-radius: 8px; margin-bottom: 15px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1);"></div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="margin-bottom: 15px;">
                        <span style="background-color: {card.get('badge_color', '#3b82f6')}; color: white; padding: 4px 8px; font-size: 0.7rem; font-weight: 800; border-radius: 4px; display: inline-block; margin-bottom: 8px;">
                            {card.get('badge', 'RECOMMENDED')}
                        </span>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #1f2937;">{card.get('card_name')}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div style='color: #4f46e5; font-size: 0.9rem; font-weight: bold; margin-bottom: 15px;'>주요 혜택</div>",
                    unsafe_allow_html=True,
                )
                for b in card.get("benefits", [])[:-1]:
                    st.markdown(
                        f"""
                    <div style="display: flex; align-items: flex-start; margin-bottom: 12px; font-size: 0.85rem;">
                        <div style="margin-right: 12px; font-size: 1.1rem;">✦</div>
                        <div style="line-height: 1.4;">
                            <span style="color: #1f2937; font-weight: 700;">{b.get('benefit_name', '')} </span>
                            <span style="color: #4b5563; font-weight: 400;"> {b.get('summary', '')}</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.write("")

                st.markdown(
                    f"""
                <div style="background-color: #f3f4f6; border-radius: 8px; padding: 15px; text-align: left; margin-bottom: 15px;">
                    <div style="color: #6b7280; font-size: 0.75rem; font-weight: bold; margin-bottom: 5px;">연회비 (전월실적)</div>
                    <div style="color: #4338ca; font-size: 1.1rem; font-weight: 800;">
                        {format_fee(card.get("fee"))} <span style="font-size: 0.85rem; color: #6b7280; font-weight: 600;">({card.get('condition', '-')})</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.link_button(
                    label=card.get("btn_text", "자세히 보기"),
                    url=card.get("detail_url", "#"),
                    use_container_width=True,
                    type="primary",
                )


def render_mindmap_tab():
    if not st.session_state.analysis_result:
        st.info(
            "아직 분석된 소비 패턴이 없습니다. 'Chat (카드 추천)' 탭에서 챗봇과 대화를 나누어 보세요!"
        )
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 내 소비 패턴 마인드맵")
        nodes = [Node(id="root", label="내 소비 패턴", size=30, color="#3b82f6")]
        edges = []
        for cat in st.session_state.analysis_result:
            node_id = f"cat_{cat['id']}"
            nodes.append(
                Node(
                    id=node_id,
                    label=f"{cat['name']}\n({cat['percent']}%)",
                    size=20,
                    color="#9333ea",
                )
            )
            edges.append(Edge(source="root", target=node_id))

        config = Config(width=700, height=550)
        clicked_id = agraph(nodes=nodes, edges=edges, config=config)

    with col2:
        st.subheader("💡 카테고리 맞춤 카드 추천")
        if clicked_id and clicked_id != "root":
            if st.session_state.last_clicked_id != clicked_id:
                st.session_state.card_index = 0
                st.session_state.last_clicked_id = clicked_id

            cat_data = next(
                (
                    c
                    for c in st.session_state.analysis_result
                    if f"cat_{c['id']}" == clicked_id
                ),
                {"id": clicked_id, "name": clicked_id},
            )
            cards = search_similar_cards_by_category(cat_data, 3)
            if cards:
                current_idx = st.session_state.card_index
                card = cards[current_idx]

                st.markdown(
                    f"<h3 style='text-align: center;'>{card.get('card_name')}</h3>",
                    unsafe_allow_html=True,
                )
                st.write("")

                nav_left, img_center, nav_right = st.columns([1, 4, 1], gap="small")
                with nav_left:
                    st.markdown("<br><br><br>", unsafe_allow_html=True)
                    if st.button(
                        "◀",
                        disabled=(current_idx == 0),
                        use_container_width=True,
                        key="prev_btn_map",
                    ):
                        st.session_state.card_index -= 1
                        st.rerun()
                with img_center:
                    bg_style = (
                        f"background-image: url('{card.get('image_url')}'); background-size: cover; background-position: center;"
                        if card.get("image_url")
                        else f"background: '#ccc');"
                    )

                    st.markdown(
                        f"""
                        <div style="{bg_style} height: 240px; border-radius: 12px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1);"></div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='text-align: center; margin-top:10px;'><b>{current_idx + 1} / {len(cards)}</b></div>",
                        unsafe_allow_html=True,
                    )
                with nav_right:
                    st.markdown("<br><br><br>", unsafe_allow_html=True)
                    if st.button(
                        "▶",
                        disabled=(current_idx == len(cards) - 1),
                        use_container_width=True,
                        key="next_btn_map",
                    ):
                        st.session_state.card_index += 1
                        st.rerun()

                st.write("---")

                for b in card.get("benefits", [])[:-1]:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #eef2ff, #e0e7ff);
                            border-radius: 12px;
                            padding: 16px 18px;
                            margin-bottom: 12px;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                        ">
                            <div style="
                                font-size: 0.85rem;
                                font-weight: 700;
                                color: #4338ca;
                                margin-bottom: 6px;
                                letter-spacing: -0.3px;
                            ">
                                ✦ {b['benefit_name']}
                            </div>
                            <div style="
                                font-size: 0.9rem;
                                color: #374151;
                                line-height: 1.5;
                                font-weight: 500;
                            ">
                                {b.get('summary', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("왼쪽 마인드맵에서 원하시는 카테고리를 클릭해보세요!")


# ==========================================
# 5. 메인 앱 실행부
# ==========================================
def main():
    st.set_page_config(
        page_title="Financial Concierge (v2 - MultiQuery RRF)", layout="wide"
    )
    init_session_state()

    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .block-container { padding-top: 1rem !important; padding-bottom: 8rem !important; max-width: 90rem; }
            .stButton button { text-align: left; }
            .stTabs [data-baseweb="tab-list"] { gap: 24px; }
            .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 0px; border-bottom: 2px solid transparent; padding-top: 10px; padding-bottom: 10px; }
            .stTabs [aria-selected="true"] { color: #1d4ed8 !important; border-bottom: 2px solid #1d4ed8 !important; font-weight: bold; }
            div[data-testid="stChatInput"] {
                position: fixed !important;
                bottom: 2rem !important;
                left: 50% !important;
                transform: translateX(-50%) !important;
                width: calc(100% - 2rem) !important;
                max-width: 75rem !important;
                z-index: 999 !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### **The Finance Curator** `[v2 · 멀티쿼리 RRF + 금액 바인딩]`")

    tab1, tab2 = st.tabs(["💬 Chat (카드 추천)", "🧠 Insights (나의 소비패턴)"])

    with tab1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "cards" in msg and msg["cards"]:
                    st.write("")
                    render_3_column_cards(msg["cards"])

        if prompt := st.chat_input("소비 성향에 대해 더 궁금한 점이 있으신가요?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("맞춤형 금융 상품을 탐색 중입니다..."):
                    # [STEP 1] 소비 패턴 추출
                    analysis_result = extract_consumption_pattern(prompt)
                    st.session_state.analysis_result = analysis_result.get(
                        "categories", []
                    )

                    # [STEP 2] benefit_keywords 필터 → card_id 선별 → 벡터 유사도 검색
                    recommended_cards, retrieved_docs, candidate_mode = (
                        search_similar_cards(analysis_result, top_k=3)
                    )

                    # [STEP 3] LLM 응답 생성
                    bot_reply, llm_prompt = generate_chat_response(
                        prompt, retrieved_docs, analysis_result.get("categories", [])
                    )

                    st.markdown(bot_reply)
                    render_3_column_cards(recommended_cards)

                    with st.expander("🔍 RAG 프로세스 상세 분석 (Debug Log)"):
                        st.markdown("### **1️⃣ 소비 패턴 추출 (Extraction)**")
                        st.info(
                            "사용자 입력에서 카테고리와 비중을 계산하고, 검색용 프로필을 생성합니다."
                        )
                        st.json(analysis_result)

                        st.markdown("---")
                        st.markdown("### **2️⃣ 멀티쿼리 구성 (Multi-Query)**")
                        st.write("**[Original Prompt]**")
                        st.caption(prompt)

                        debug_cats = analysis_result.get("categories", [])
                        debug_profile = analysis_result.get("search_profile", "")
                        rank_labels = ["1순위", "2순위", "3순위", "4순위"]
                        debug_queries = [
                            f"[{rank_labels[i] if i < 4 else f'{i+1}순위'} {cat['name']} {cat['percent']}%] "
                            f"{cat['name']} 집중 할인 혜택 카드  (weight={cat['percent']})"
                            for i, cat in enumerate(debug_cats)
                        ]
                        if debug_cats:
                            top_cats = debug_cats[:2]
                            top_emphasis = ", ".join(
                                f"{c['name']}({c['percent']}%) 집중 할인"
                                for c in top_cats
                            )
                            top_percent = max(c["percent"] for c in debug_cats)
                            integrated = f"{top_emphasis}. {debug_profile}. {top_cats[0]['name']} 할인이 가장 중요."
                            debug_queries.append(
                                f"{integrated}  (weight={top_percent})"
                            )

                        st.write(f"**[멀티쿼리 목록 ({len(debug_queries)}개)]**")
                        for q in debug_queries:
                            st.code(q, language="text")
                        st.caption(
                            "💡 카테고리별 비중 강조 쿼리 + 통합 쿼리로 RRF 스코어링"
                        )

                        st.markdown("---")
                        st.markdown("### **3️⃣ 메타데이터 필터링 (Hard Filter)**")
                        debug_cats = analysis_result.get("categories", [])
                        top1_kw = get_search_keywords(debug_cats[:1])
                        top2_kw = (
                            get_search_keywords(debug_cats[1:2])
                            if len(debug_cats) > 1
                            else []
                        )

                        if candidate_mode == "intersection":
                            st.success(
                                "후보 풀: **교집합** (1·2순위 키워드 모두 보유 카드)"
                            )
                        elif candidate_mode == "union":
                            st.warning(
                                "후보 풀: **합집합** 폴백 (교집합 카드 부족 → 전체 키워드 OR)"
                            )
                        else:
                            st.info("후보 풀: 키워드 없음 → 전체 검색")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**1순위 키워드 ({len(top1_kw)}개):**")
                            st.caption(", ".join(top1_kw) if top1_kw else "없음")
                        with col_b:
                            st.write(f"**2순위 키워드 ({len(top2_kw)}개):**")
                            st.caption(", ".join(top2_kw) if top2_kw else "없음")

                        st.markdown("---")
                        st.markdown("### **4️⃣ 검색 결과 (멀티쿼리 RRF 스코어링)**")
                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**최종 추천 카드 (Top 3)**")
                            for i, c in enumerate(recommended_cards):
                                st.write(
                                    f"{i+1}. {c['card_name']} ({c['card_company']})"
                                )
                        with cols[1]:
                            st.write("**컨텍스트 소스 (Chunks)**")
                            st.write(f"총 {len(retrieved_docs)}개의 텍스트 조각 참조")
                            for doc in retrieved_docs:
                                st.caption(
                                    f"📄 {doc.metadata.get('card_name')} ({len(doc.page_content)}자)"
                                )

                        st.markdown("---")
                        st.markdown("### **5️⃣ 최종 LLM 프롬프트 (금액 바인딩 적용)**")
                        with st.container(border=True):
                            st.code(llm_prompt, language="text")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": bot_reply,
                            "cards": recommended_cards,
                        }
                    )

    with tab2:
        render_mindmap_tab()


if __name__ == "__main__":
    main()
