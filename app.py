import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from streamlit_agraph import agraph, Node, Edge, Config

# Vecotor DB 청킹, 임베딩, 생성 관련은 vector_db.py 파일로 따로 관리
# LLM 설정 및 Chat 관련만 병합 진행

# ==========================================
# 1. 초기 설정 및 환경 변수
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# 수파베이스 관련 키, URL 모두 삭제
# 수파베이스 DB 대신하여 크로마DB로 생성한 벡터DB 경로 참조
VECTOR_STORE_DIR = "./VectorStores_Card"

# 크로마DB에서 선언한 Embedding과 Vectordb로 변경
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

# ── BM25 (키워드 기반) 리트리버 ──────────────────────────────────────────
# ChromaDB에서 전체 문서를 로드해 BM25 인덱스 생성
# BM25는 "교육비", "식비" 같은 정확한 키워드 매칭에 강점
_raw = vectordb.get()
_all_docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(_raw["documents"], _raw["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(_all_docs, k=15)

# ── Hybrid Retriever: BM25(0.4) + Chroma Dense(0.6) ─────────────────────
# BM25: 키워드 매칭에 강함 / Chroma: 의미적 유사도에 강함
# 두 결과를 앙상블해 상호 보완 → Cohere가 최종 top 6 재정렬
base_retriever = vectordb.as_retriever(search_kwargs={"k": 15})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, base_retriever],
    weights=[0.4, 0.6],
)
my_rerank = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-v3.5", top_n=6)
cohere_retriever = ContextualCompressionRetriever(
    base_compressor=my_rerank, base_retriever=ensemble_retriever
)


# ==========================================
# 2. 세션 상태 (Session State) 초기화
# ==========================================
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
        # 마인드맵을 그리기 위한 카테고리 데이터
        st.session_state.analysis_result = []
    if "card_index" not in st.session_state:
        # 마인드맵 탭에서의 카드 스와이프 인덱스
        st.session_state.card_index = 0
    if "last_clicked_id" not in st.session_state:
        st.session_state.last_clicked_id = None
    if "chat_history" not in st.session_state:
        # LangChain ChatMessageHistory 객체를 병합하였습니다.
        # generate_chat_response()에서 대화 맥락을 LLM에 전달하기 위해 사용
        # Streamlit의 session_state에 보관하여 대화 기록을 유지함
        st.session_state.chat_history = ChatMessageHistory()


# ==========================================
# 3. AI 및 DB 로직 (RAG, LLM 함수)
# ==========================================
def extract_consumption_pattern(user_text):
    """
    사용자 대화에서 소비 패턴을 추출하여 JSON 형태로 반환 (마인드맵 용도)
    """
    # [실제 적용 시 주석 해제]
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """사용자의 소비 패턴을 분석하여 JSON 형식으로 응답하세요.
                반드시 'categories'라는 키에 [id, label, percent]를 포함한 리스트를 담아야 합니다.

                예:
                {{"categories": [
                    {{"id": "food", "label": "식비", "percent": 40}},
                    {{"id": "transport", "label": "교통", "percent": 20}},
                    {{"id": "shopping", "label": "온라인쇼핑", "percent": 20}},
                ]}}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = extract_prompt | chat_model
    response = chain.invoke({"question": user_text})
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    if not clean_json:
        return []
    try:
        return json.loads(clean_json)["categories"]
    except (json.JSONDecodeError, KeyError):
        return []

    # # [Mock Data: UI 테스트용]
    # return [
    #     {"id": "food", "label": "배달/외식", "percent": 50},
    #     {"id": "transport", "label": "대중교통", "percent": 30},
    #     {"id": "shopping", "label": "온라인쇼핑", "percent": 20},
    # ]


def format_fee(fee):
    if not fee or fee == "-":
        return "정보 없음"
    if fee == "없음":
        return "무료"
    try:
        return f"{int(fee):,}원"
    except:
        return fee + "원"  # 이미 문자열이면 그대로


# [신규 추가 함수]
# 기존에는 Supabase RPC가 UI에 필요한 모든 필드를 직접 반환했음.
# ChromaDB는 Document 객체를 반환하므로, metadata에서 필드를 꺼내
# render_3_column_cards()가 기대하는 dict 형태로 재조립하는 변환 레이어가 필요해짐.
# → 두 검색 함수(search_similar_cards, search_similar_cards_by_category)가
#   공통으로 사용하도록 별도 함수로 분리함.
def _docs_to_cards(docs):
    """ChromaDB Document 리스트를 UI 렌더링용 카드 dict list로 변환"""

    # 청킹으로 인해 같은 카드가 여러 Document로 쪼개져 있을 수 있음
    # card_id 기준으로 중복을 제거하여 카드당 1개만 반환
    seen = set()
    cards = []
    for doc in docs:
        meta = doc.metadata
        card_id = meta.get("card_id", "")
        if card_id in seen:
            continue
        seen.add(card_id)

        # benefits_json: vector_db.py에서 JSON 문자열로 직렬화한 혜택 리스트
        # → 역직렬화하여 [{benefit_name, summary}] 형태로 복원
        try:
            benefits = json.loads(meta.get("benefits_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            benefits = []

        # render_3_column_cards()가 기대하는 키명으로 매핑
        # metadata 키명(vector_db.py 기준) → UI 키명(render 함수 기준)
        #   annual_fee_domestic → fee
        #   base_performance    → condition
        #   badge / badge_color / btn_text 는 cards.json에 없으므로 기본값 고정
        cards.append(
            {
                "card_name": meta.get("card_name", "-"),
                "card_company": meta.get("card_company", "-"),
                "image_url": meta.get("image_url", ""),
                "detail_url": meta.get("detail_url", "#"),
                "fee": meta.get("annual_fee_domestic", "-"),
                "condition": meta.get("base_performance", "-"),
                "benefits": benefits,
                "badge": "RECOMMENDED",  # Vector DB에 기대하는 값이 없음으로 하드 코딩함
                "badge_color": "#3b82f6",  # Vector DB에 기대하는 값이 없음으로 하드 코딩함
                "btn_text": "자세히 보기",  # Vector DB에 기대하는 값이 없음으로 하드 코딩함
            }
        )
    return cards


# [기존 함수 교체]
# 변경 전: get_embedding()으로 직접 임베딩 → supabase.rpc("match_cards") 호출
# 변경 후: vectordb.similarity_search()로 ChromaDB에서 직접 유사도 검색
#          결과를 _docs_to_cards()로 변환하여 반환
# k=top_k*2 로 여유있게 검색 후 슬라이싱하는 이유:
#   청킹으로 같은 카드가 중복 등장할 수 있어 중복 제거 후 top_k가 모자랄 수 있기 때문
def search_similar_cards(query, top_k=3):
    """Hybrid(BM25+Chroma) → Cohere 리랭킹 기반 유사 카드 검색
    반환: (cards, docs)
      - cards: UI 렌더링용 dict 리스트 (metadata 기반)
      - docs:  LLM context용 Document 리스트 (page_content 포함)
    """
    try:
        docs = cohere_retriever.invoke(query)
        cards = _docs_to_cards(docs)[:top_k]
        # UI에 표시될 카드 이름 기준으로 관련 doc만 필터링
        card_names = {c["card_name"] for c in cards}
        filtered_docs = [d for d in docs if d.metadata.get("card_name") in card_names]
        return cards, filtered_docs

    except Exception as e:
        st.error(f"카드 검색 중 오류: {e}")
        return [], []


# [기존 함수 교체]
# 변경 전: get_embedding()으로 직접 임베딩 → supabase.rpc("match_cards") 호출
# 변경 후: category['label']로 자연어 쿼리를 구성해 vectordb.similarity_search() 호출
#          마인드맵 노드 클릭 시 해당 카테고리에 맞는 카드를 검색하는 용도
def search_similar_cards_by_category(category, top_k=3):
    """카테고리 레이블 기반 유사 카드 검색"""
    try:
        query_text = f"{category['label']} 관련 소비 혜택 카드"

        docs = vectordb.similarity_search(query_text, k=top_k * 2)
        return _docs_to_cards(docs)[:top_k]

    except Exception as e:
        st.error(f"카드 검색 중 오류: {e}")
        return []


# [수정] cards 파라미터 추가
# 기존: 함수 내부에서 독립적으로 similarity_search 실행
#   → LLM이 참조하는 카드와 하단에 렌더링되는 카드가 달라지는 문제 발생
# 변경: main()에서 search_similar_cards()로 먼저 검색한 뒤 결과를 이 함수에 전달
#   → LLM context와 렌더링 카드가 동일한 데이터를 바라보도록 보장
def generate_chat_response(user_text, docs):
    """RAG 기반 챗봇 응답 생성
    - docs: page_content 기반 풍부한 context (혜택 상세 조건 포함)
    """
    # docs의 page_content를 카드별로 그룹핑해 전체 혜택 상세까지 LLM에 전달
    # 기존 방식(metadata 요약 5개)보다 훨씬 풍부한 정보 제공
    card_contents: dict[str, list[str]] = {}
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        card_contents.setdefault(name, []).append(doc.page_content)

    context = "\n\n---\n\n".join("\n".join(chunks) for chunks in card_contents.values())

    formatted_history = ""
    for msg in st.session_state.chat_history.messages:
        role = "사용자" if msg.type == "human" else "AI"
        formatted_history += f"{role}: {msg.content}\n"

    prompt = f"""당신은 카드 추천 전문가입니다.
제공된 [추천 카드 목록]을 바탕으로 아래 [사고 과정]에 따라 논리적으로 분석하여 답변하세요.

[사고 과정(Chain-of-Thought)]
1. 사용자의 주요 소비 패턴(영화, 카페, 쇼핑 등)과 요구사항을 파악한다.
2. 카드 데이터 중 해당 요구사항에 가장 부합하는 혜택을 가진 카드를 분석한다.
3. 선정한 카드들이 사용자에게 실질적으로 어떤 이득을 주는지 논리적 근거를 도출한다.
4. 최종적으로 서로 다른 카드 TOP 3를 선정하여 결과를 출력한다.

[답변 규칙]
- 반드시 서로 다른 카드 TOP 3를 순서대로 추천할 것.
- 문서에 없는 내용은 절대 지어내지 말 것.
- 텍스트 출력 시 "카드사"는 제외합니다.
- 답변은 아래 [출력 형식]을 엄격히 따를 것.

[과거 대화 기록]
{formatted_history}

[추천 카드 목록]
{context}

[사용자 질문]
{user_text}

-- 출력 형식 --
#### **카드 이름 1**\n
**주요 혜택명** : 세부 사항 요약\n
- 추천 이유 설명 문장

#### **카드 이름 2**\n
**주요 혜택명** : 세부 사항 요약\n
- 추천 이유 설명 문장

#### **카드 이름 3**\n
**주요 혜택명** : 세부 사항 요약\n
- 추천 이유 설명 문장
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
    """채팅 창 내 3열 카드 출력 UI"""
    if not cards:
        return

    cols = st.columns(3)
    for i, card in enumerate(cards[:3]):
        with cols[i]:
            with st.container(border=True):
                # 1. 카드 상단 이미지 (텍스트 겹침 제거, URL 지원)
                bg_style = (
                    f"background-image: url('{card.get('image_url')}'); background-size: cover; background-position: center;"
                    if card.get("image_url")
                    else "background: #ccc;"
                )

                st.markdown(
                    f"""
                    <div style="{bg_style} height: 160px; border-radius: 8px; margin-bottom: 15px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1);"></div>
                """,
                    unsafe_allow_html=True,
                )

                # 2. 뱃지 및 카드명 (이미지 하단으로 분리)
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

                # 3. 주요 혜택 리스트 (아이콘 포함)
                st.markdown(
                    "<div style='color: #4f46e5; font-size: 0.9rem; font-weight: bold; margin-bottom: 15px;'>주요 혜택</div>",
                    unsafe_allow_html=True,
                )
                # benefits_json 구조: [{"benefit_name": "카테고리명", "summary": "요약 텍스트"}]
                # 수정함 : benefit_name(볼드), summary → 일반 텍스트로 시인성 고려해서 요약본을 출력하는 것으로 변경함
                # 기존에는 혜택 상세 내용이 출력되었음
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

                # 4. 연회비 및 전월실적 박스 (수정됨)
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

                # 5. 하단 링크 버튼 (외부 URL로 이동)
                st.link_button(
                    label=card.get("btn_text", "자세히 보기"),
                    url=card.get("detail_url", "#"),
                    use_container_width=True,
                    type="primary",
                )


def render_mindmap_tab():
    """Insights (나의 소비패턴) 탭 렌더링"""
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
            nodes.append(
                Node(
                    id=cat["id"],
                    label=f"{cat['label']}\n({cat['percent']}%)",
                    size=20,
                    color="#9333ea",
                )
            )
            edges.append(Edge(source="root", target=cat["id"]))

        config = Config(
            width="100%",
            height=550,
            directed=False,
            physics={"enabled": True, "stabilization": {"iterations": 200}},
        )
        clicked_id = agraph(nodes=nodes, edges=edges, config=config)

    with col2:
        st.subheader("💡 카테고리 맞춤 카드 추천")
        if clicked_id and clicked_id != "root":
            if st.session_state.last_clicked_id != clicked_id:
                st.session_state.card_index = 0
                st.session_state.last_clicked_id = clicked_id

            # 기존 코드: search_similar_cards(clicked_id, 3)
            #   → clicked_id는 "food", "transport" 같은 영문 문자열을 그대로 검색 쿼리로 사용
            #
            # 변경 후: search_similar_cards_by_category(cat_data, 3)
            #   → 함수 내부에서 category['label']로 자연어 쿼리를 구성하므로
            #     영문대신 한국어 label(예: "배달/외식")이 담긴 dict가 필요함
            #   → analysis_result에서 clicked_id와 일치하는 카테고리 dict를 찾아 전달하여 정확도 향상

            cat_data = next(
                (c for c in st.session_state.analysis_result if c["id"] == clicked_id),
                {"id": clicked_id, "label": clicked_id},
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
                    # gradient 필드는 cards.json에 존재하지 않으며 추가 계획도 없습니다.
                    # 카드별 배경 색상이 필요하다면 card 데이터에 의존하지 않고
                    # 별도의 색상 매핑 dict(예: CARD_COLORS = {card_id: gradient})를 만들어
                    # card.get('gradient') 대신 CARD_COLORS.get(card_id, '#ccc') 방식으로 구현이 필요해보입니다.
                    bg_style = (
                        f"background-image: url('{card.get('image_url')}'); background-size: cover; background-position: center;"
                        if card.get("image_url")
                        else f"background: {card.get('gradient', '#ccc')};"
                    )

                    st.markdown(
                        f"""
                        <div style="{bg_style} height: 200px; border-radius: 12px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1);"></div>
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
                    st.info(f"#### {b['benefit_name']}")
        else:
            st.info("왼쪽 마인드맵에서 원하시는 카테고리를 클릭해보세요!")


# ==========================================
# 5. 메인 앱 실행부
# ==========================================
def main():
    # 사이드바를 사용하지 않으므로 initial_sidebar_state 설정을 제거했습니다.
    st.set_page_config(page_title="Financial Concierge", layout="wide")
    init_session_state()

    # CSS 스타일링
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
                max-width: 75rem !important; /* 위쪽 채팅 컨텐츠 너비와 동일하게 맞춤 */
                z-index: 999 !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # 기존의 render_sidebar() 호출부 삭제 완료

    st.markdown("### **The Finance Curator**")

    # 상단 탭 구성
    tab1, tab2 = st.tabs(["💬 Chat (카드 추천)", "🧠 Insights (나의 소비패턴)"])

    # [TAB 1] 챗봇
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
                    categories = extract_consumption_pattern(prompt)
                    st.session_state.analysis_result = categories

                    # [STEP 2] 검색 쿼리 재구성 (노이즈 제거)
                    if categories:
                        search_query = (
                            " ".join([c["label"] for c in categories])
                            + " 할인 혜택 카드"
                        )
                    else:
                        search_query = prompt

                    # [STEP 3] Hybrid(BM25+Chroma) → Cohere Rerank
                    recommended_cards, retrieved_docs = search_similar_cards(
                        search_query, 3
                    )

                    # [STEP 4] LLM 응답 생성 (page_content 기반 풍부한 context)
                    bot_reply, llm_prompt = generate_chat_response(
                        prompt, retrieved_docs
                    )

                    st.markdown(bot_reply)
                    render_3_column_cards(recommended_cards)

                    # 디버그 로그 (접기 가능)
                    with st.expander("🔍 디버그 로그"):
                        st.markdown("**① 추출된 소비 패턴 카테고리**")
                        st.json(categories)
                        st.markdown("**② 벡터 검색 쿼리**")
                        st.code(search_query, language="text")
                        st.markdown("**③ 검색된 카드**")
                        st.write([c["card_name"] for c in recommended_cards])
                        st.markdown(
                            f"**④ LLM context 문서 수: {len(retrieved_docs)}개 chunks**"
                        )
                        for doc in retrieved_docs:
                            st.caption(
                                f"📄 {doc.metadata.get('card_name')} — {len(doc.page_content)}자"
                            )
                        st.markdown("**⑤ LLM에 전달된 전체 프롬프트**")
                        st.code(llm_prompt, language="text")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": bot_reply,
                            "cards": recommended_cards,
                        }
                    )

    # [TAB 2] 마인드맵
    with tab2:
        render_mindmap_tab()


if __name__ == "__main__":
    main()
