import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

def ask_llm(context, query):
    prompt = f"""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question:
{query}

Answer clearly and concisely based on the context.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "nvidia/nemotron-3-nano-30b-a3b:free",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()['choices'][0]['message']['content']