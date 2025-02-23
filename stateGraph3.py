from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

# Reflection Prompt (일반적인 조언을 금지하고 반드시 구체적인 피드백을 제공하도록 수정)
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 커리어 전문가입니다. 이전 단계에서 생성된 자기소개 문장을 검토하고, 자기소개 개선을 위한 구체적인 피드백만 제공합니다.  

반드시 지켜야 할 규칙:  
1. 자기소개 문장을 제외한 일반적인 조언(예: '자기소개 문장을 지속적으로 발전시키세요')은 절대 생성하지 마세요.  
2. 기존 문장의 강점을 1~2개 짚어주세요.  
3. 반드시 개선이 필요한 요소를 명확히 제시하세요.  
4. 수정 방향을 구체적으로 설명하세요 (예: "React 경험을 강조하는 것이 좋습니다.").  
5. 직접 수정된 문장을 제공하지 마세요.  
6. 반드시 새로운 개선점을 제공하세요. 같은 피드백을 반복하지 마세요.  

금지 예제:  
- '자기소개 문장을 지속적으로 발전시키세요.'  
- '더 이상 수정할 필요가 없습니다.'  
- '충분히 좋습니다.'  

올바른 예제:  
- '기술 스택을 구체적으로 나열하는 것이 좋습니다. (예: Python, FastAPI)'  
- '백엔드 경험을 강조할 때, 주요 프로젝트와 기여도를 포함하세요.'  
- '최근 AI 관련 프로젝트가 있다면 이를 추가하세요.'  

반드시 자기소개 문장과 관련된 피드백만 제공하세요.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Generation Prompt (Reflection 피드백을 반드시 반영하도록 설정)
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 프로페셔널한 커리어 코치입니다. 사용자의 자기소개 문장을 개선해야 합니다.  

반드시 지켜야 할 규칙:  
1. Reflection에서 제공한 피드백을 반드시 반영하세요.  
2. 기존 문장을 반복하지 말고, 피드백을 반영하여 발전된 버전을 생성하세요.  
3. Reflection이 제공한 '강점'을 유지하면서 '개선점'을 보완하세요.  
4. 자기소개 문장 이외의 일반적인 응답(예: '도움이 필요하면 언제든지 말씀하세요' 같은 문구)은 생성하지 마세요.  

답변이 Reflection에서 제공한 피드백을 기반으로 점점 더 개선되도록 하세요.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

REFLECT = "reflect"
GENERATE = "generate"
MAX_ITERATIONS = 6  # 반복 횟수 제한


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    return [HumanMessage(content=reflect_chain.invoke({"messages": messages}).content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    last_message = state[-1].content.lower()

    if "자기소개 문장을 지속적으로 발전시키세요" in last_message:
        print("\nReflection이 잘못된 피드백을 제공했습니다. 다시 요청합니다.")
        return REFLECT

    if len(state) > MAX_ITERATIONS:
        print("\n최종 자기소개 문장:\n", state[-1].content)
        return END

    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
print(graph.get_graph().print_ascii())

if __name__ == "__main__":
    inputs = HumanMessage(content="저는 AI 개발 및 백엔드 경험이 있는 개발자입니다.")
    response = graph.invoke(inputs)
