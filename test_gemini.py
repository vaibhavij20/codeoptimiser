from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("\n" + "=" * 60)
print("TEST GEMINI DEBUG")
print("API KEY PREFIX :", api_key[:15])
print("MODEL          : gemini-2.5-flash")
print("=" * 60 + "\n")

client = genai.Client(
    api_key=api_key
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello"
)

print("Response:")
print(response.text)