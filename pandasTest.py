import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd
import tiktoken

# .env 파일에서 환경 변수 로드
load_dotenv()

# GPT-4o-mini 설정
gpt4o_mini = ChatOpenAI(
    model_name="gpt-4o-mini",  # GPT-4o-mini
    temperature=0.7,
    max_tokens=150,
)

# GPT-4o 설정
gpt4o = ChatOpenAI(
    model_name="gpt-4o",  # GPT-4o
    temperature=0.7,
    max_tokens=300,
)

file_path = "data/CARD_SUBWAY_MONTH_202311.csv"
df = pd.read_csv(file_path, encoding="utf-8")
# CSV 데이터를 문자열로 변환
data = df
# print(data)


def encoding_getter(encoding_type: str):
    try:
        return tiktoken.encoding_for_model(encoding_type)
    except KeyError:
        print(
            f"Warning: {encoding_type} is not recognized. Defaulting to 'cl100k_base'."
        )
        return tiktoken.get_encoding("cl100k_base")


def tokenizer(string: str, encoding_type: str) -> list:
    encoding = encoding_getter(encoding_type)
    print(encoding)
    tokens = encoding.encode(string)
    return tokens


def token_counter(string: str, encoding_type: str) -> int:
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens


prompt_text = f"다음 데이터를 요약하고 정리해줘:\n{data}"

models = ["GPT-4o-mini", "GPT-4o"]

for model in models:
    num_tokens = token_counter(prompt_text, model)
    print(f"{model} : {str(num_tokens)}")


# # GPT-4o-mini 사용
response_mini = gpt4o_mini.invoke(
    [HumanMessage(content=f"다음 데이터를 요약하고 정리해줘:\n{data}")]
)
print(response_mini.content)

# # GPT-4o 사용
# response_full = gpt4o.invoke(
#     [HumanMessage(content=f"다음 데이터를 요약하고 정리해줘:\n{data}")]
# )
# print(response_full.content)
