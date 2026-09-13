"""
app.py - AI Data Analyst Streamlit UI
Upload any CSV → AI analyzes it → generates charts + insights
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv

from analyst_agent import get_best_model, get_data_summary, generate_analysis_code, generate_summary_report
from code_executor import execute_analysis, clean_code

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Upload any CSV → AI writes Python → generates charts + insights · Powered by Groq (FREE)")

with st.expander("ℹ️ How it works"):
    st.markdown("""
**3 steps, fully automated:**

| Step | What happens |
|------|--------------|
| 1️⃣ Upload | You upload any CSV file |
| 2️⃣ AI writes code | The AI reads your data and writes custom Python analysis code |
| 3️⃣ Run & report | The code runs automatically → charts appear + a plain-English report |

**Works with any CSV:** sales data, survey results, stock prices, sports stats, etc.
**Powered by Groq (FREE)** — 14,400 requests/day, no credit card needed.
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Free key at console.groq.com — no credit card needed!"
    )
    st.success("✅ Free tier\n14,400 requests/day\nNo credit card needed")

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
- Upload any CSV file
- Ask a specific question for focused analysis
- Works best with 50–10,000 rows
- Numeric + date columns get the richest charts
    """)

    st.markdown("---")
    st.markdown("### 📥 Sample datasets")
    st.markdown("No CSV? Try a free one:")
    st.markdown("- [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)")
    st.markdown("- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)")
    st.markdown("- [Sales](https://raw.githubusercontent.com/dsrscientist/dataset1/master/sales_data_sample.csv)")

# ── Main area ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

question = st.text_input(
    "💬 Optional: Ask a specific question about your data",
    placeholder="e.g. Which product has the highest sales? What trend do you see over time?"
)

analyze_btn = st.button("🚀 Analyze Data", type="primary", disabled=(uploaded_file is None))

if uploaded_file and not analyze_btn:
    # Preview the data
    try:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset for later use
        st.markdown("### 👀 Data Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df_preview.shape[0]:,}")
        col2.metric("Columns", df_preview.shape[1])
        col3.metric("Numeric columns", len(df_preview.select_dtypes(include="number").columns))
        st.dataframe(df_preview.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

if analyze_btn:
    if not api_key:
        st.error("❌ Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
        st.stop()

    # Load the data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read CSV file: {e}")
        st.stop()

    st.markdown("---")
    st.subheader(f"🔬 Analyzing: **{uploaded_file.name}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric columns", len(df.select_dtypes(include="number").columns))

    # Step indicators
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("#### 🧠 Understanding data")
        box1 = st.empty()
        box1.info("⏳ Waiting...")
    with s2:
        st.markdown("#### ✍️ Writing analysis code")
        box2 = st.empty()
        box2.info("⏳ Waiting...")
    with s3:
        st.markdown("#### ▶️ Running code")
        box3 = st.empty()
        box3.info("⏳ Waiting...")

    progress = st.progress(0, text="Starting...")

    try:
        # Step 1: Get model + data summary
        box1.warning("🔄 Running...")
        progress.progress(10, text="Detecting best Groq model...")
        model = get_best_model(api_key)
        st.caption(f"Using model: `{model}`")

        progress.progress(20, text="Summarizing your data...")
        data_summary = get_data_summary(df)
        box1.success("✓ Done")
        progress.progress(33, text="Data understood!")

        # Step 2: Generate analysis code
        box2.warning("🔄 AI writing code...")
        progress.progress(40, text="AI is writing custom analysis code...")

        raw_code = generate_analysis_code(api_key, model, data_summary, question)
        code = clean_code(raw_code)
        box2.success("✓ Code written")
        progress.progress(60, text="Code ready!")

        # Show the generated code (collapsible)
        with st.expander("🔍 View generated Python code"):
            st.code(code, language="python")

        # Step 3: Execute the code
        box3.warning("🔄 Running analysis...")
        progress.progress(70, text="Running code and generating charts...")

        with tempfile.TemporaryDirectory() as tmpdir:
            exec_result = execute_analysis(code, df, work_dir=tmpdir)

            if not exec_result["success"]:
                box3.error("⚠️ Code had errors")
                st.warning("⚠️ The generated code had an error. Showing partial results.")
                with st.expander("Error details"):
                    st.code(exec_result["error"])
            else:
                box3.success("✓ Analysis complete")

            progress.progress(85, text="Generating report...")

            # Generate the plain-English report
            insights = exec_result.get("insights", [])
            report = generate_summary_report(api_key, model, data_summary, insights, question)

            progress.progress(100, text="✅ All done!")

            # ── Results ────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📋 Analysis Report")
            st.markdown(report)

            # Show charts
            chart_files = exec_result.get("charts", [])
            if chart_files:
                st.markdown("---")
                st.markdown("## 📈 Charts")
                # Display in grid: 2 per row
                for i in range(0, len(chart_files), 2):
                    cols = st.columns(2)
                    for j, chart_path in enumerate(chart_files[i:i+2]):
                        with cols[j]:
                            st.image(chart_path, use_container_width=True)
            else:
                st.info("No charts were generated. Try a dataset with more numeric columns.")

            # Show raw insights if available
            if insights:
                st.markdown("---")
                with st.expander("📊 Raw computed insights"):
                    for ins in insights:
                        st.markdown(f"- {ins}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
