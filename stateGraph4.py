import os

# import base64
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import pandas as pd

# import pytesseract
from tika import parser
from groq import Groq

# 환경변수 로드
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# 상태 타입 정의: 파일 경로, 파일 분류, 처리된 텍스트, 요약 결과
class FileState(TypedDict):
    file_path: str
    file_type: Optional[str]  # "image", "pdf", "csv"
    processed_text: Optional[str]
    summary: Optional[str]


# 1단계: 파일명을 보고 분류하는 노드
def classify_file(state: FileState) -> FileState:
    file_path = state["file_path"]
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        state["file_type"] = "image"
    elif ext == ".pdf":
        state["file_type"] = "pdf"
    elif ext == ".csv":
        state["file_type"] = "csv"
    return state


# 2-1: 이미지 파일 처리 (OCR, 한글 인식)
# def process_image(state: FileState) -> FileState:
#     file_path = state["file_path"]
#     image = Image.open(file_path)
#     text = pytesseract.image_to_string(image, lang="kor")
#     state["processed_text"] = text
#     return state


# llama-3.2-90b-vision-preview 모델 사용
# def process_image(state: FileState) -> FileState:
#     file_path = state["file_path"]
#     # 이미지 파일을 바이너리로 읽고 base64로 인코딩
#     with open(file_path, "rb") as f:
#         image_data = f.read()
#     image_base64 = base64.b64encode(image_data).decode("utf-8")

#     # 이미지 분석 프롬프트
#     prompt = f"다음 이미지를 분석하여 텍스트를 추출해줘:\n{image_base64}"

#     # llama-3.2-90b-vision-preview 모델 호출
#     chat_completion = groq_client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.2-90b-vision-preview",
#     )
#     extracted_text = chat_completion.choices[0].message.content
#     state["processed_text"] = extracted_text
#     return state


def process_image(state: FileState) -> FileState:
    file_path = state["file_path"]
    from paddleocr import PaddleOCR

    # PaddleOCR 초기화 (한국어 지원, 기울기 보정 포함)
    ocr = PaddleOCR(use_angle_cls=True, lang="korean")
    result = ocr.ocr(file_path, cls=True)

    # 결과가 없는 경우 처리
    if not result or not result[0]:
        print("OCR 결과가 없습니다.")
        state["processed_text"] = ""
        return state

    # OCR 텍스트 추출
    text_parts = []
    for line in result[0]:
        try:
            rec_text = line[1][0]
            text_parts.append(rec_text)
        except Exception as e:
            print(f"Skipping invalid OCR line: {line} - {e}")

    # OCR 텍스트를 하나의 문자열로 저장
    state["processed_text"] = " ".join(text_parts)

    # OCR 결과 출력
    print("Extracted text:")
    print(state["processed_text"])

    return state


# 2-2: PDF 파일 처리
def process_pdf(state: FileState) -> FileState:
    file_path = state["file_path"]
    parsed = parser.from_file(file_path)
    text = parsed.get("content", "")
    state["processed_text"] = text.strip() if text else ""
    return state


# 2-3: CSV 파일 처리
def process_csv(state: FileState) -> FileState:
    file_path = state["file_path"]
    df = pd.read_csv(file_path, encoding="utf-8")
    state["processed_text"] = df
    return state


# 3단계: LLM 활용해 요약하는 노드
def summarize(state: FileState) -> FileState:
    data = state["processed_text"]
    prompt = f"다음 데이터를 요약하고 정리해줘:\n{data}"
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gemma2-9b-it",
    )
    state["summary"] = chat_completion.choices[0].message.content
    return state


# 상태 그래프 생성
workflow = StateGraph(FileState)

# 노드 등록
workflow.add_node("classify", classify_file)
workflow.add_node("process_image", process_image)
workflow.add_node("process_pdf", process_pdf)
workflow.add_node("process_csv", process_csv)
workflow.add_node("summarize", summarize)

# START에서 분류 노드로 연결
workflow.add_edge(START, "classify")

# 분류 노드에서 조건부 에지를 사용해 각 분기 처리
workflow.add_conditional_edges(
    "classify",
    lambda state: state["file_type"],
    {
        "image": "process_image",
        "pdf": "process_pdf",
        "csv": "process_csv",
    },
)

# 각 분기 노드에서 요약 노드로 에지 연결
workflow.add_edge("process_image", "summarize")
workflow.add_edge("process_pdf", "summarize")
workflow.add_edge("process_csv", "summarize")

# 요약 노드에서 END로 연결
workflow.add_edge("summarize", END)

# 그래프 컴파일 (executor 생성)
executor = workflow.compile()

# 그래프 다이어그램 출력 (Mermaid 및 ASCII)
print(executor.get_graph().draw_mermaid())
executor.get_graph().print_ascii()

# 파일 경로 입력
# file_path = "data/2023년 인구성장률 현황.pdf"
# file_path = "data/2023년 혼인건수 현황.png"
# file_path = "data/CARD_SUBWAY_MONTH_202311.csv"
# file_path = "data/2023년 인구성장률 현황.csv"
file_path = "data/test.png"

initial_state: FileState = {
    "file_path": file_path,
    "file_type": None,
    "processed_text": None,
    "summary": None,
}
result = executor.invoke(initial_state)
print("\nGemma2 요약 결과:")
print(result["summary"])
