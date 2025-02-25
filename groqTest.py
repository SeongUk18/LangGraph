import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "한국어 자기소개 해봐봐",
        }
    ],
    model="gemma2-9b-it",
)

print(chat_completion.choices[0].message.content)
