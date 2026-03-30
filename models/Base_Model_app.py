import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

# Base Vector Similarity 베이스 모델
# 커스텀 필터링 로직 없이 사용자 입력을 그대로 ChromaDB에 쿼리

# ==========================================
# 1. 초기 설정 및 환경 변수
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_DIR = "./VectorStores_Card"

embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


# ==========================================
# 2. 세션 상태 초기화
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
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()


# ==========================================
# 3. RAG 핵심 로직
# ==========================================
def search_similar_cards(user_query, top_k=3):
    """사용자 입력을 그대로 ChromaDB에 쿼리 → 유사도 상위 청크 → top_k 카드 반환"""
    docs = vectordb.similarity_search(user_query, k=top_k * 3)
    cards = _docs_to_cards(docs)[:top_k]
    card_ids = {c["card_id"] for c in cards}
    filtered_docs = [d for d in docs if d.metadata.get("card_id") in card_ids]
    return cards, filtered_docs


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
                "card_id": card_id,
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


def generate_chat_response(user_text, docs):
    card_contents: dict[str, list[str]] = {}
    for doc in docs:
        name = doc.metadata.get("card_name", "")
        cleaned = re.sub(
            r"\[유의사항\].*?(?=\n\[|\Z)", "", doc.page_content, flags=re.DOTALL
        ).strip()
        if cleaned:
            card_contents.setdefault(name, []).append(cleaned)

    context_parts = []
    for card_name, chunks in card_contents.items():
        card_section = f"【 카드명: {card_name} 】\n" + "\n\n".join(chunks)
        context_parts.append(card_section)
    context = "\n\n---\n\n".join(context_parts)

    card_names = list(card_contents.keys())
    card_name_1 = card_names[0] if len(card_names) > 0 else "카드 1"
    card_name_2 = card_names[1] if len(card_names) > 1 else "카드 2"
    card_name_3 = card_names[2] if len(card_names) > 2 else "카드 3"

    prompt = f"""당신은 카드 추천 전문가입니다.
아래 [추천 카드 목록]은 사용자에게 이미 화면에 보여주는 카드들입니다.
반드시 이 카드들을 기준으로, 각 카드가 왜 사용자의 질문에 적합한지 구체적으로 설명하세요.

[추천 카드 목록]
{context}

[사용자 질문]
{user_text}

[답변 규칙]
1. 반드시 아래 카드 3개를 모두 순서대로 추천하세요.
   - {card_name_1}
   - {card_name_2}
   - {card_name_3}
2. 각 카드별로 사용자 질문과 관련된 혜택 2가지를 선정하세요.
3. 추천 이유는 카드 혜택과 사용자 질문을 연결하여 2~3문장으로 작성하세요.

-- 출력 형식 --
#### [**{card_name_1}**]
**혜택명1**: 혜택 요약 설명 1줄
**혜택명2**: 혜택 요약 설명 1줄

- 추천 이유: 2~3문장 설명

#### [**{card_name_2}**]
**혜택명1**: 혜택 요약 설명 1줄
**혜택명2**: 혜택 요약 설명 1줄

- 추천 이유: 2~3문장 설명

#### [**{card_name_3}**]
**혜택명1**: 혜택 요약 설명 1줄
**혜택명2**: 혜택 요약 설명 1줄

- 추천 이유: 2~3문장 설명
"""

    response = chat_model.invoke(prompt)
    reply = response.content

    st.session_state.chat_history.add_user_message(user_text)
    st.session_state.chat_history.add_ai_message(reply)

    return reply


# ==========================================
# 4. UI 컴포넌트
# ==========================================
def format_fee(fee):
    if not fee or fee == "-":
        return "정보 없음"
    if fee == "없음":
        return "무료"
    try:
        return f"{int(fee):,}원"
    except:
        return fee + "원"


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


# ==========================================
# 5. 메인 앱 실행부
# ==========================================
def main():
    st.set_page_config(
        page_title="Financial Concierge (v3 - Pure Similarity)", layout="wide"
    )
    init_session_state()

    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .block-container { padding-top: 1rem !important; padding-bottom: 8rem !important; max-width: 90rem; }
            .stButton button { text-align: left; }
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

    st.markdown("### **The Finance Curator** `[v3 · Pure Vector Similarity]`")

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
                recommended_cards, retrieved_docs = search_similar_cards(
                    prompt, top_k=3
                )
                bot_reply = generate_chat_response(prompt, retrieved_docs)

                st.markdown(bot_reply)
                render_3_column_cards(recommended_cards)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": bot_reply,
                        "cards": recommended_cards,
                    }
                )


if __name__ == "__main__":
    main()
