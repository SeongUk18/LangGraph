import openai
import json
import chromadb
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="processed_data")

# 1. 데이터
data = [
    {"date": "2024-01-01", "sales": 1200, "region": "서울"},
    {"date": "2024-01-02", "sales": 1500, "region": "서울"},
    {"date": "2024-01-01", "sales": 1100, "region": "부산"},
    {"date": "2024-01-02", "sales": 1700, "region": "부산"},
]

df = pd.DataFrame(data)


# 2. LLM을 이용한 데이터 전처리
def process_data_with_llm(data):
    json_data = json.dumps(data, ensure_ascii=False, indent=2)

    prompt = f"""
    다음 데이터에서 중요한 정보만 정리해주세요:

    {json_data}

    - 'date'와 'region'을 기준으로 묶어서 총 매출을 계산.
    - 모든 날짜와 지역의 데이터를 유지해야 하며, 특정 날짜가 누락되지 않도록 정리.
    - 반환 형식: [{{"date": "YYYY-MM-DD", "region": "지역명", "total_sales": number}}]
    """

    response = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "system", "content": prompt}], temperature=0
    )

    return json.loads(response.choices[0].message.content)


processed_data = process_data_with_llm(df.to_dict(orient="records"))


# 3. LLM이 정리한 데이터를 벡터화하여 ChromaDB에 저장
def get_embedding(text):
    """텍스트 데이터를 벡터로 변환"""
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


# 데이터베이스 저장
collection.add(
    ids=[str(i) for i in range(len(processed_data))],
    embeddings=[
        get_embedding(json.dumps(item, ensure_ascii=False)) for item in processed_data
    ],
    metadatas=processed_data,
)


# 4. 적절한 데이터 검색
def search_data_with_llm(query):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding], n_results=5  # 유사한 5개 데이터 검색
    )

    return results["metadatas"][0]  # 검색된 데이터 반환


# 5. 검색된 데이터 시각화
def visualize_data(query):
    """검색된 데이터를 그래프로 시각화"""
    data = search_data_with_llm(query)

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["date"], df["total_sales"], marker="o", linestyle="-", label="Total Sales"
    )
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Trend")
    plt.legend()
    plt.grid()
    plt.show()


# 6. 실행 예시
visualize_data("2024년 서울 1월의 총 매출을 보여줘")
