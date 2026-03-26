# ==========================================
# 청킹 + 임베딩 + ChromaDB 생성 (1회 실행용)
# ==========================================
import json
import os
import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

MY_API_KEY = os.getenv("OPENAI_API_KEY")

# 파일 목록 정의
json_files = ["data/cards.json"]

# Recursive Splitter 객체 생성
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=50, separators=["\n\n", "\n", " "]
)

my_chunks = []


# 혜택 키워드 추출 함수: benefits_structured의 benefits_name 고유값 반환
def extract_benefit_keywords(benefits_structured):
    return list(
        {
            b.get("benefits_name", "")
            for b in benefits_structured
            if b.get("benefits_name")
        }
    )


# JSON 데이터 로드 및 Document 객체 변환
for file_name in json_files:
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

        for card in data:
            benefits_structured = card.get("benefits_structured", [])

            # 혜택 요약: "카테고리명: 요약" 형태로 결합 (검색 쿼리와의 유사도 매칭용)
            if benefits_structured:
                benefits = " / ".join(
                    [
                        f"{b.get('benefits_name', '')}: {b.get('benefits_summary', '')}"
                        for b in benefits_structured
                    ]
                )
            else:
                benefits = str(card)

            # 혜택 상세: benefits_details를 page_content에 포함시켜 검색 정확도 향상
            # (예: "이마트에서만 적용", "월 5회 한도" 같은 세부 조건도 유사도 검색에 반영됨)
            # benefits_summary만 임베딩하면 세부 조건 검색 시 매칭이 누락될 수 있음
            # 원문 텍스트는 page_content에 보존되어 RAG 응답 시 LLM context로 전달됨
            if benefits_structured:
                details_lines = []
                for b in benefits_structured:
                    name = b.get("benefits_name", "")
                    detail = b.get("benefits_details", "").strip()
                    if detail:
                        details_lines.append(f"[{name}]\n{detail}")
                details_text = "\n\n".join(details_lines)
            else:
                details_text = ""

            # 카드 종류 구분
            if "travel" in file_name.lower():
                card_group = "travel"
            else:
                card_group = "general"

            # 혜택 키워드 추출 (benefits_name 고유값)
            benefit_keywords = extract_benefit_keywords(benefits_structured)
            benefit_keywords_str = ", ".join(benefit_keywords)

            # annual_fee 처리 (domestic/overseas 구조 대응)
            annual_fee = card.get("annual_fee", {})
            if isinstance(annual_fee, dict):
                annual_fee_domestic = annual_fee.get("domestic", "-")
                annual_fee_str = f"국내 {annual_fee_domestic}원 / 해외 {annual_fee.get('overseas', '-')}원"
            else:
                annual_fee_domestic = str(annual_fee)
                annual_fee_str = annual_fee_domestic

            # UI 렌더링용 혜택 리스트 직렬화 (상위 5개만)
            benefits_for_ui = [
                {
                    "benefit_name": b.get("benefits_name", ""),
                    "summary": b.get("benefits_summary", ""),
                }
                for b in benefits_structured[:5]
            ]
            benefits_json_str = json.dumps(benefits_for_ui, ensure_ascii=False)

            # AI가 읽을 전체 내용 구성
            # 혜택요약 → 빠른 매칭, 혜택상세 → 세부 조건까지 커버
            # splitter가 chunk_size 기준으로 분할하므로 텍스트가 길어도 문제없음
            content = f"""
카드명: {card['card_name']}
카드사: {card['card_company']}
카드종류: {card_group}
전월실적: {card.get('base_performance', '')}
연회비: {annual_fee_str}
주요혜택: {benefits}
혜택카테고리: {benefit_keywords_str}
혜택상세: {details_text}
            """.strip()

            # LangChain에서 인식하는 Document 객체로 생성
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_name,
                    "card_id": card.get("card_id", ""),
                    "card_name": card["card_name"],
                    "card_company": card["card_company"],
                    "card_group": card_group,
                    "benefit_keywords": benefit_keywords_str,
                    # UI 렌더링에 필요한 추가 메타데이터
                    "image_url": card.get("image_url", ""),
                    "detail_url": card.get("detail_url", ""),
                    "base_performance": card.get("base_performance", "-"),
                    "annual_fee_domestic": annual_fee_domestic,
                    "benefits_json": benefits_json_str,
                },
            )

            # 생성된 문서를 splitter를 이용해 청킹하여 리스트에 추가
            my_chunks.extend(splitter.split_documents([doc]))

print("총 청크 개수 :", len(my_chunks))
print("-" * 30)
if my_chunks:
    print("첫 번째 청크 예시:\n", my_chunks[0].page_content)
    print("첫 번째 청크 메타데이터:\n", my_chunks[0].metadata)

# 임베딩 설정
my_embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=MY_API_KEY)

# 벡터 DB 생성 및 로컬 저장
my_directory = "./VectorStores_Card"

# 배치 단위로 분할하여 TPM 한도 초과 방지
BATCH_SIZE = 50  # 한 번에 처리할 청크 수
MAX_RETRIES = 5

def add_documents_with_retry(vectordb, docs):
    for attempt in range(MAX_RETRIES):
        try:
            vectordb.add_documents(docs)
            return
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                wait_sec = 20 * (attempt + 1)
                print(f"  Rate limit 초과. {wait_sec}초 대기 후 재시도 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_sec)
            else:
                raise

# 첫 번째 배치로 DB 초기화
first_batch = my_chunks[:BATCH_SIZE]
vectordb = Chroma.from_documents(
    documents=first_batch,
    embedding=my_embedding,
    persist_directory=my_directory,
    collection_metadata={"hnsw:space": "cosine"},
)
print(f"배치 1/{(len(my_chunks) - 1) // BATCH_SIZE + 1} 완료")

# 나머지 배치 추가
for i in range(BATCH_SIZE, len(my_chunks), BATCH_SIZE):
    batch = my_chunks[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(my_chunks) - 1) // BATCH_SIZE + 1
    print(f"배치 {batch_num}/{total_batches} 처리 중...")
    add_documents_with_retry(vectordb, batch)
    time.sleep(1)  # 배치 간 짧은 대기

print(f"성공. '{my_directory}' 폴더에 카드 데이터 벡터 DB가 생성되었습니다.")
