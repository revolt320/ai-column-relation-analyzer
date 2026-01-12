import streamlit as st
import pandas as pd
from openai import OpenAI
import json

# ---------- CONFIG ----------
st.set_page_config(page_title="Batch Category Classifier", layout="wide")
st.title("Column Relationship Checker")

client = OpenAI()
BATCH_SIZE = 5

# ---------- AI BATCH FUNCTION ----------
def classify_batch(pairs):
    formatted_pairs = "\n".join(
        [f"{i+1}. Category: {c} | Search Category: {s}"
         for i, (c, s) in enumerate(pairs)]
    )

    prompt = f"""
Classify if each category falls within its search category.

For each item return:
- match: "Yes" or "No"
- score: integer from 0 to 100 indicating relevance strength

Return ONLY valid JSON in this format:
[
  {{"match": "Yes", "score": 85}}
]

Items:
{formatted_pairs}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict classifier. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)


# ---------- SIDEBAR (INPUT) ----------
st.sidebar.header("Input Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()

    category_col = st.sidebar.selectbox(
        "Select CATEGORY column",
        columns
    )

    search_col = st.sidebar.selectbox(
        "Select SEARCH CATEGORY column",
        columns,
        index=1 if len(columns) > 1 else 0
    )

    run = st.sidebar.button("Run AI Classification")

    # ---------- MAIN AREA (OUTPUT) ----------
    st
