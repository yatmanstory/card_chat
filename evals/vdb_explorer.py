import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

load_dotenv()

st.set_page_config(page_title="VectorDB Explorer", layout="wide")
st.title("🗄️ ChromaDB Explorer")

VECTOR_STORE_DIR = "./VectorStores_Card"


@st.cache_resource
def load_vectordb():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.list_collections()[0]
    raw = client.get_collection(collection.name)
    return vectordb, raw


vectordb, raw_collection = load_vectordb()

tab1, tab2, tab3, tab4 = st.tabs(
    ["① DB 구조", "② 자연어 검색", "③ 혜택명 검색", "④ 카드ID 검색"]
)

# ==========================================
# TAB 1: 청크 수 + 메타데이터 구조
# ==========================================
with tab1:
    st.subheader("DB 구조 확인")
    if st.button("조회하기", key="btn_structure"):
        st.metric("총 청크 수", raw_collection.count())

        sample = raw_collection.get(limit=1, include=["documents", "metadatas"])
        meta = sample["metadatas"][0]

        st.markdown("#### 메타데이터 필드 구조")
        rows = [{"필드명": k, "샘플 값": str(v)[:80]} for k, v in meta.items()]
        st.table(rows)

        st.markdown("#### page_content 샘플 (첫 번째 청크)")
        st.code(sample["documents"][0][:500], language="text")

# ==========================================
# TAB 2: 자연어 유사도 검색
# ==========================================
with tab2:
    st.subheader("자연어 유사도 검색")
    query = st.text_input(
        "검색어를 입력하세요", placeholder="예: 해외여행 마일리지 공항 라운지"
    )
    k = st.slider("반환 청크 수", min_value=1, max_value=10, value=3)

    if st.button("검색", key="btn_semantic"):
        if query:
            docs = vectordb.similarity_search(query, k=k)
            for i, doc in enumerate(docs):
                with st.expander(f"결과 {i+1} — {doc.metadata.get('card_name', '-')}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("**메타데이터**")
                        st.json(
                            {
                                "card_id": doc.metadata.get("card_id"),
                                "card_name": doc.metadata.get("card_name"),
                                "card_company": doc.metadata.get("card_company"),
                                "benefit_keywords": doc.metadata.get(
                                    "benefit_keywords"
                                ),
                            }
                        )
                    with col2:
                        st.markdown("**page_content**")
                        st.code(doc.page_content, language="text")

# ==========================================
# TAB 3: benefit_keywords 필터 검색
# ==========================================
with tab3:
    st.subheader("혜택명(benefit_keywords)으로 카드 검색")
    benefit = st.text_input("혜택명 입력", placeholder="예: 배달앱, 카페")
    mode = st.radio(
        "검색 모드",
        options=["AND (모두 포함)", "OR (하나라도 포함)"],
        horizontal=True,
        help="AND: 입력한 키워드를 모두 보유한 카드 / OR: 하나라도 보유한 카드",
    )

    if st.button("검색", key="btn_benefit"):
        if benefit:
            kw_list = [k.strip() for k in benefit.split(",") if k.strip()]
            use_and = mode.startswith("AND") and len(kw_list) > 1

            if len(kw_list) == 1:
                chroma_filter = {"benefit_keywords": {"$contains": kw_list[0]}}
            elif use_and:
                chroma_filter = {
                    "$and": [{"benefit_keywords": {"$contains": kw}} for kw in kw_list]
                }
            else:
                chroma_filter = {
                    "$or": [{"benefit_keywords": {"$contains": kw}} for kw in kw_list]
                }

            results = raw_collection.get(
                where=chroma_filter, include=["metadatas", "documents"]
            )

            seen: set = set()
            cards = []
            for meta, doc in zip(results["metadatas"], results["documents"]):
                cid = meta.get("card_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    cards.append((meta, doc))

            op = " AND " if use_and else " OR "
            label = op.join(f'"{k}"' for k in kw_list)
            st.info(f"{label} 혜택을 가진 카드: **{len(cards)}개**")
            for meta, doc in cards:
                with st.expander(
                    f"{meta.get('card_name')} ({meta.get('card_company')})"
                ):
                    st.markdown(f"- **card_id**: `{meta.get('card_id')}`")
                    st.markdown(
                        f"- **benefit_keywords**: {meta.get('benefit_keywords')}"
                    )
# ==========================================
# TAB 4: card_id로 청크 조회
# ==========================================
with tab4:
    st.subheader("card_id로 관련 청크 전체 조회")
    card_id_input = st.text_input("card_id 입력", placeholder="예: card_001")

    if st.button("검색", key="btn_cardid"):
        if card_id_input:
            results = raw_collection.get(
                where={"card_id": {"$eq": card_id_input}},
                include=["metadatas", "documents"],
            )
            metadatas = results["metadatas"]
            documents = results["documents"]

            if not documents:
                st.warning(f"'{card_id_input}'에 해당하는 청크가 없습니다.")
            else:
                card_name = metadatas[0].get("card_name", "-")
                st.success(f"**{card_name}** — 총 {len(documents)}개 청크")
                for i, (meta, doc) in enumerate(zip(metadatas, documents)):
                    with st.expander(f"청크 {i+1}"):
                        st.code(doc, language="text")
