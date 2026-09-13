"""
diagnose.py - Run this to find out which Gemini models work with your API key.
    python diagnose.py
"""

import requests
import os

# Try to get key from .env file first
api_key = ""
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()

if not api_key:
    api_key = input("Enter your Gemini API key: ").strip()

print(f"\n🔑 Testing API key: {api_key[:8]}...{api_key[-4:]}\n")

# ── Step 1: List all available models ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: Listing available models...")
print("=" * 60)

resp = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    params={"key": api_key},
    timeout=15
)

if resp.status_code != 200:
    print(f"❌ FAILED to list models: {resp.status_code}")
    print(resp.text)
    print("\n⚠️  Your API key may be invalid. Please get a new one at:")
    print("   https://aistudio.google.com/apikey")
    exit()

models = resp.json().get("models", [])
chat_models = [m["name"] for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
print(f"✅ Found {len(chat_models)} models that support generateContent:\n")
for m in chat_models:
    print(f"   {m}")

# ── Step 2: Test the most common models ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Testing which models respond to a simple question...")
print("=" * 60)

test_models = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro",
    "gemini-pro",
    "gemini-1.0-pro",
]

working_model = None
for model_name in test_models:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with just the word: WORKING"}]}],
        "generationConfig": {"maxOutputTokens": 10}
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=15)
    if resp.status_code == 200:
        print(f"✅ {model_name} — WORKS!")
        if not working_model:
            working_model = model_name
    else:
        print(f"❌ {model_name} — {resp.status_code}: {resp.json().get('error', {}).get('message', '')[:80]}")

# ── Result ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if working_model:
    print(f"✅ RESULT: Use model '{working_model}'")
    print(f"\nI'll now update your rag_engine.py and reranker.py to use '{working_model}'...")

    for fname in ["rag_engine.py", "reranker.py"]:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                content = f.read()
            # Replace any gemini model name with the working one
            for m in test_models:
                content = content.replace(f'"{m}"', f'"{working_model}"')
                content = content.replace(f"/{m}:", f"/{working_model}:")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"   ✅ Updated {fname}")

    print(f"\n🎉 Done! Now run: streamlit run app.py")
else:
    print("❌ No working models found.")
    print("   Your API key may be invalid or your region is blocked.")
    print("   Please create a new API key at: https://aistudio.google.com/apikey")
