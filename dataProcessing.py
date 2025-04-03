import os
import pandas as pd
import tiktoken
import json
from dotenv import load_dotenv
from groq import Groq

# 환경 변수 로드
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 모델별 최대 토큰 제한 (입력 + 출력 합산)
MODEL_MAX_TOKENS = 15000
SAFETY_MARGIN = 11000
MAX_TOKENS = MODEL_MAX_TOKENS - SAFETY_MARGIN  # 실제 사용 가능한 입력 토큰 수 (4000)


# 토큰 계산 함수
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# Groq API 호출 (응답을 반환)
def send_to_groq(batch_text):
    print(f"\n LLM에 배치 전송 중... (토큰 수: {count_tokens(batch_text)})")
    print(f"요청 데이터 일부:\n{batch_text[:500]}...\n")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    아래는 한국 지하철 이용 통계 데이터입니다.
                    json
                    [
                        {{"노선명": "2호선", "역명": "강남", "승차총승객수": 150000, "하차총승객수": 145000}},
                        {{"노선명": "1호선", "역명": "서울역", "승차총승객수": 120000, "하차총승객수": 118000}}
                    ]
                    
                    **반드시 위와 같은 JSON 리스트([]) 형식으로 변환하세요.**

                    **반드시 JSON 리스트([]) 형식으로만 출력하세요.**
                    **추가 설명, 메타데이터, 불필요한 텍스트를 절대 포함하지 마세요.**
                    **다른 형식(예: Markdown, CSV 등)으로 변환하지 마세요.**

                    다음 데이터를 변환하세요:

                    {batch_text}
                    """.strip(),
                }
            ],
            model="gemma2-9b-it",
        )

        # 응답 처리
        if chat_completion.choices and chat_completion.choices[0].message:
            response_text = chat_completion.choices[0].message.content.strip()
            print(f"응답 데이터 일부:\n{response_text[:500]}...\n")
            return response_text
    except Exception as e:
        print(f"오류 발생: {e}")
    return ""


# CSV 데이터를 변환하여 LLM에 전송 
def process_batches(file_path: str):
    df = pd.read_csv(file_path)
    data_list = df.to_dict(orient="records")
    json_text = json.dumps(data_list, ensure_ascii=False, indent=2)

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(json_text)

    # 전체 배치 처리
    all_results = []
    for i in range(0, len(tokens), MAX_TOKENS):
        batch_tokens = tokens[i : i + MAX_TOKENS]
        batch_text = encoding.decode(batch_tokens)
        batch_result = send_to_groq(batch_text)
        if batch_result:
            all_results.append(batch_result)

    print("전체 데이터 처리 완료.")
    return all_results


# 실행
results = process_batches("data/CARD_SUBWAY_MONTH_202311.csv")

for idx, res in enumerate(results[:10]):
    print(f"결과 {idx + 1} 일부:\n{res[:500]}...\n")
