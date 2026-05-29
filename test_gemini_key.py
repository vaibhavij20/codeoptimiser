"""Test if the Gemini API key is valid."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"Testing API key: {api_key[:20]}...")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[{"role": "user", "parts": [{"text": "Say 'API key is working'"}]}],
    )
    print(f"✅ API key is valid!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ API key failed: {e}")
