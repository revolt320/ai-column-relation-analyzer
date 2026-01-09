import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from collections import defaultdict

# ---------- CONFIG ----------
st.set_page_config(page_title="Batch Category Classifier", layout="centered")
st.title("Column relationship checker")

client = OpenAI()
BATCH_SIZE = 5

# ---------- AI BATCH FUNCTION ----------
def classify_batch(pairs):
    """
    pairs: list of tuples [(category, search_category), ...]
    returns: list of dicts [{match: Yes/No, score: int}]
    """

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


# ---------- UI ----------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    category_col = st.selectbox("Select CATEGORY column", columns)
    search_col = st.selectbox("Select SEARCH CATEGORY column", columns, index=1 if len(columns) > 1 else 0)

    if st.button("Run AI Classification"):
        cache = {}
        results_match = []
        results_score = []

        rows = list(df[[category_col, search_col]].itertuples(index=False, name=None))

        with st.spinner("Running batch AI classification..."):
            i = 0
            while i < len(rows):
                batch = rows[i:i + BATCH_SIZE]

                uncached = []
                uncached_indices = []

                for idx, pair in enumerate(batch):
                    if pair not in cache:
                        uncached.append(pair)
                        uncached_indices.append(idx)

                if uncached:
                    ai_results = classify_batch(uncached)
                    for pair, res in zip(uncached, ai_results):
                        cache[pair] = res

                for pair in batch:
                    res = cache[pair]
                    results_match.append(res["match"])
                    results_score.append(res["score"])

                i += BATCH_SIZE

        df["ai_match"] = results_match
        df["ai_relevance_score"] = results_score

        st.success("Classification completed with batching + caching")
        st.dataframe(df.head())

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Result CSV",
            data=csv,
            file_name="classified_output.csv",
            mime="text/csv"
        )
