from ollama import chat
from openai import OpenAI

client = OpenAI(
    api_key="lm-studio",
    base_url="http://127.0.0.1:1234/v1"
)


response = client.chat.completions.create(
    model = "local-model",
    messages=[
        {
            "role": "user",
            "content": "Giải thích Machine Learning đơn giản dễ hiểu"
        }
    ],
    temperature=0.0
)

print(response.choices[0].message.content)