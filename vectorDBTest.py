import openai
import json
import chromadb
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime, timezone, timedelta

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="processed_data")


# 날짜를 타임스탬프로 변환하는 함수
def date_to_timestamp(date_str):
    return int(time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple()))


# 데이터 저장 시 변환
original_data = [
    {"date": "2025-01-01", "region": "서울", "total_sales": 1200},
    {"date": "2025-01-02", "region": "서울", "total_sales": 1500},
    {"date": "2025-01-01", "region": "부산", "total_sales": 1100},
    {"date": "2025-01-02", "region": "부산", "total_sales": 1700},
]

df = pd.DataFrame(original_data)


# LLM을 이용한 데이터 전처리
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

# LLM이 반환한 데이터 확인
print("LLM이 정리한 데이터:", processed_data)

# 날짜를 타임스탬프로 변환하여 저장
for item in processed_data:
    if isinstance(item["date"], str):  # 문자열이면 변환
        item["date"] = date_to_timestamp(item["date"])

print("최종 저장된 데이터 확인:", processed_data)


# 벡터 변환 함수
def get_embedding(text):
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


# ChromaDB에 데이터 저장
collection.add(
    ids=[str(i) for i in range(len(processed_data))],
    embeddings=[
        get_embedding(json.dumps(item, ensure_ascii=False)) for item in processed_data
    ],
    metadatas=processed_data,
)

print("데이터 저장 완료")
print("저장된 메타데이터 확인:")
print(collection.get())


# LLM을 사용하여 자연어 쿼리를 분석하고 유동적인 필터를 생성
def extract_query_conditions(query):
    prompt = f"""
    사용자의 검색 요청을 분석하여 적절한 데이터베이스 검색 조건을 JSON 형태로 변환하세요.

    가능한 입력 예시:
    - "서울 2025년 1월의 총 매출을 보여줘" → {{"region": "서울", "date_range": ["2025-01-01", "2025-01-31"]}}
    - "부산의 2025년 1월 2일 매출을 알려줘" → {{"region": "부산", "date": "2025-01-02"}}
    - "2025년 1월 1일 모든 지역의 매출을 비교해줘" → {{"date": "2025-01-01"}}
    - "서울의 매출을 보여줘" → {{"region": "서울"}}
    - "2025년 1월의 매출을 보여줘" → {{"date_range": ["2025-01-01", "2025-01-31"]}}

    JSON 형식으로 정확히 변환해야 합니다:
    {{
        "region": "서울",
        "date": "2025-01-01",
        "date_range": ["2025-01-01", "2025-01-31"]
    }}

    사용자의 입력: "{query}"
    
    JSON 결과:
    """

    response = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "system", "content": prompt}], temperature=0
    )

    return json.loads(response.choices[0].message.content)


# LLM을 사용하여 검색 수행
def search_data_with_llm(query):
    query_conditions = extract_query_conditions(query)
    print("LLM이 생성한 검색 조건:", query_conditions)

    query_embedding = get_embedding(query)
    filter_conditions = []

    if "region" in query_conditions:
        filter_conditions.append({"region": query_conditions["region"]})

    if "date" in query_conditions:
        converted_date = date_to_timestamp(query_conditions["date"])
        filter_conditions.append({"date": converted_date})

    if "date_range" in query_conditions:
        start_date, end_date = query_conditions["date_range"]
        converted_start = date_to_timestamp(start_date)
        converted_end = date_to_timestamp(end_date)
        print("적용된 검색 필터 date_range:", converted_start, "-", converted_end)
        filter_conditions.append({"date": {"$gte": converted_start}})
        filter_conditions.append({"date": {"$lte": converted_end}})

    filter_query = {"$and": filter_conditions} if filter_conditions else {}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=filter_query,
    )

    metadata = (
        sum(results["metadatas"], [])
        if isinstance(results["metadatas"][0], list)
        else results["metadatas"]
    )
    print("검색된 데이터:", metadata)

    return metadata


# 타임스탬프를 날짜 문자열(YYYY-MM-DD)로 변환하는 함수
def timestamp_to_date(timestamp):
    kst = timezone(timedelta(hours=9))  # 한국 표준시 (UTC+9)
    return datetime.fromtimestamp(timestamp, tz=kst).strftime("%Y-%m-%d")


# 검색된 데이터를 그래프로 시각화
def visualize_data(query):
    data = search_data_with_llm(query)

    if not data:
        print("검색된 데이터가 없습니다.")
        return

    for item in data:
        item["date"] = timestamp_to_date(item["date"])

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


# 실행
visualize_data("2025년 서울 1월의 총 매출을 보여줘")
