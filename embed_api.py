from dotenv import load_dotenv
import os
from google import genai

load_dotenv(".env")

# code for Gemeni
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
result = client.models.embed_content(
    model = "gemini-embedding-2",
    contents = "Giải thích cho tôi Machine Learning là gì bằng tiếng Việt trong 3 câu"
)

print(result.embeddings[0])


