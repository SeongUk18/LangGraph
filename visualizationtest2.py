import pandas as pd
import matplotlib.pyplot as plt
import openai
import seaborn as sns
import os
from load_dotenv import load_dotenv

load_dotenv()

# ✅ 예시 데이터 생성: 부서별 연봉 및 근속 연수
data = {
    "Department": ["HR", "HR", "HR", "Sales", "Sales", "Sales", "Dev", "Dev", "Dev"],
    "YearsAtCompany": [2, 5, 7, 1, 3, 6, 1, 4, 8],
    "Salary": [40000, 55000, 60000, 38000, 50000, 58000, 45000, 62000, 72000],
}
df = pd.DataFrame(data)


# ✅ Step 1: 데이터프레임 요약
def summarize_dataframe(df: pd.DataFrame) -> str:
    summary = "다음은 데이터프레임의 컬럼 정보입니다:\n"
    for col in df.columns:
        dtype = df[col].dtype
        examples = df[col].dropna().unique()[:3]
        summary += f"- {col} (type: {dtype}, 예시: {examples})\n"
    return summary


# ✅ Step 2: LLM에게 시각화 코드 요청
def ask_llm_for_plot_code(df_summary: str) -> str:
    prompt = f"""
다음은 데이터프레임의 요약입니다:

{df_summary}

이 데이터를 시각화할 수 있는 적절한 matplotlib 또는 seaborn 코드를 한 가지 생성해주세요.
코드는 한 개의 차트를 그리고 plt.show()로 결과를 보여주기만 하면 됩니다.
데이터는 'df'라는 변수명으로 이미 로드되어 있다고 가정해주세요.
파이썬 코드를 제외하고는 다른 말은 하지마세요.
코드 블록에 넣을 필요없어요.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.3
    )
    return response.choices[0].message.content


# ✅ 실행
summary = summarize_dataframe(df)
code_str = ask_llm_for_plot_code(summary)

print("🔍 생성된 코드:\n")
print(code_str)

# ✅ 생성된 시각화 코드 실행
exec(code_str)
