from ollama import chat

response = chat(
    model='qwen3.5:2b',
    messages=[{'role': 'user', 'content': 'Hãy giới thiệu về các thành phố lớn ở Việt Nam bằng tiếng Việt trong 3 câu'}],
)

for chunk in response:
    print(chunk['message']['content'], end='', flush=True)