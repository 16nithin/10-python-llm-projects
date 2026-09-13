"""
Run from inside your rag_system folder:
    python fix_model.py
Changes gemini-1.5-flash → gemini-2.5-flash in all code files.
"""
import os

files_to_fix = ["rag_engine.py", "reranker.py"]
old = "gemini-1.5-flash"
new = "gemini-2.5-flash"

for fname in files_to_fix:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
        updated = content.replace(old, new)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(updated)
        count = content.count(old)
        print(f"✅ {fname} — replaced {count} occurrence(s)")
    else:
        print(f"⚠️  {fname} not found")

print("\n🎉 Done! Now run: streamlit run app.py")
print("   Upload a document, ask a question — it will work!")
