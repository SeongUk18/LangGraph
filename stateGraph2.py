from typing import Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
import re

# 날씨 데이터 (날씨 상태 + 온도)
weather_data = {
    "New York": ("Sunny", "25°C"),
    "London": ("Rainy", "15°C"),
    "Tokyo": ("Cloudy", "20°C"),
    "Paris": ("Partly cloudy", "22°C"),
}


# 상태 클래스
class QueryState(TypedDict):
    query: str
    location: Optional[str]
    response: Optional[str]


# 질문을 분류하는 함수
def classify_query(state: QueryState):
    query = state["query"].lower()

    if "weather" in query and "in" in query:
        return {"query_type": "weather"}
    elif "temperature" in query and "in" in query:
        return {"query_type": "temperature"}
    else:
        return {"query_type": "unknown"}


# 위치 추출 함수
def extract_location(query: str) -> Optional[str]:
    words = query.split()
    if "in" in words:
        index = words.index("in")
        if index + 1 < len(words):
            location = words[index + 1]
            if index + 2 < len(words) and words[index + 2][0].isupper():
                location += " " + words[index + 2]
            location = re.sub(r"[^\w\s]", "", location)  # 문장부호 제거
            return location
    return None


# 날씨 상태 응답 생성
def get_weather_response(state: QueryState):
    location = state["location"]
    if location in weather_data:
        weather, _ = weather_data[location]  # 날씨 상태만 가져오기
        return {"response": f"The weather in {location} is {weather}."}
    return {"response": "Weather data not available."}


# 온도 응답 생성
def get_temperature_response(state: QueryState):
    location = state["location"]
    if location in weather_data:
        _, temperature = weather_data[location]  # 온도만 가져오기
        return {"response": f"The current temperature in {location} is {temperature}."}
    return {"response": "Temperature data not available."}


# 일반적인 질문에 대한 기본 응답
def handle_unknown_query(state: QueryState):
    return {"response": "I'm not sure. Please ask about weather or temperature."}


# 상태 그래프 생성
workflow = StateGraph(QueryState)

# 노드 추가
workflow.add_node("classify_query", classify_query)
workflow.add_node("get_weather", get_weather_response)
workflow.add_node("get_temperature", get_temperature_response)
workflow.add_node("handle_unknown", handle_unknown_query)

# 시작 지점 설정
workflow.add_edge(START, "classify_query")

# 조건부 에지 추가
workflow.add_conditional_edges(
    "classify_query",
    lambda state: state["query_type"],
    {
        "weather": "get_weather",
        "temperature": "get_temperature",
        "unknown": "handle_unknown",
    },
)


# 그래프 실행
executor = workflow.compile()

# 다양한 질문을 테스트
test_queries = [
    "What is the weather like in New York?",
    "What is the temperature in Tokyo?",
    "Can you tell me the weather in Paris?",
    "How hot is it in London?",
    "Is it raining in Tokyo?",
    "Tell me the temperature in Seoul.",
]

for query in test_queries:
    result = executor.invoke(
        {"query": query, "location": extract_location(query), "response": None}
    )
    print(f"Query: {query}")
    print(f"Response: {result['response']}")
    print("=" * 40)

print(executor.get_graph().draw_mermaid())
executor.get_graph().print_ascii()
