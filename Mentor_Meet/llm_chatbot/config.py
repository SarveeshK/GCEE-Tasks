import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

API_TOKEN = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-3.5-super-vl:free"

SYSTEM_PROMPT = """
You are a simple, helpful AI chatbot.

Rules:
- Answer in 3 to 5 short sentences by default.
- Use simple English.
- Give examples only when useful.
- Use bullet points only when necessary.
- Avoid long introductions and unnecessary details.
- If the user asks for a detailed explanation, provide more detail.
"""

MAX_RESPONSE_TOKENS = 1500
TEMPERATURE = 0.7