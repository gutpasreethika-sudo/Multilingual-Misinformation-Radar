import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

#PAGE CONFIG 
st.set_page_config(
    page_title="Multilingual Misinformation Radar",
    page_icon="🚨",
    layout="wide"
)

#  PATHS 
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "Data" / "misinformation_posts.csv"
MODEL_PATH = BASE_DIR / "Models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "Models" / "vectorizer.pkl"

#  LOAD DATA 
df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

#TITLE
st.title("🚨 Multilingual Misinformation Radar")

st.write(
    "Analyze multilingual social media posts and detect misinformation."
)

# METRICS 
st.subheader("📊 Project Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Posts", len(df))
c2.metric("Fake Posts", len(df[df["label"] == "Fake"]))
c3.metric("Real Posts", len(df[df["label"] == "Real"]))
c4.metric("Misleading Posts", len(df[df["label"] == "Misleading"]))

st.divider()

#  DATASET 
st.subheader("📄 Dataset Preview")
st.dataframe(df)

st.divider()

# CHARTS 
st.subheader("📈 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Label Distribution")
    st.bar_chart(df["label"].value_counts())

with col2:
    st.write("### Language Distribution")
    st.bar_chart(df["language"].value_counts())

st.divider()

# LANGUAGE FILTER
st.subheader("🌐 Filter By Language")

language = st.selectbox(
    "Choose Language",
    ["All"] + sorted(df["language"].unique())
)

if language == "All":
    st.dataframe(df)
else:
    st.dataframe(df[df["language"] == language])

st.divider()

# PREDICTION
st.subheader("🧠 Check a New Post")

user_post = st.text_area(
    "Enter a social media post:"
)

if st.button("Analyze Post"):

    if user_post.strip() == "":
        st.warning("Please enter a post.")
    else:

        try:

            vector = vectorizer.transform([user_post])

            prediction = model.predict(vector)[0]

            st.write("Prediction Value:", prediction)

            # HANDLE BOTH TYPES OF MODELS
            if str(prediction).lower() in ["fake", "1"]:

                st.error("🚨 FAKE NEWS DETECTED")

            elif str(prediction).lower() in ["real", "0"]:

                st.success("✅ REAL NEWS")

            elif str(prediction).lower() == "misleading":

                st.warning("⚠️ MISLEADING CONTENT")

            else:

                st.info(f"Model Output: {prediction}")

        except Exception as e:

            st.error(f"Prediction Error: {e}")

st.divider()

# INSIGHTS
st.subheader("📌 Key Insights")

st.write("""
- Fake posts often contain misleading claims and suspicious links.
- Dataset contains English, Hindi and Telugu content.
- Language filtering helps analyze misinformation trends.
- Machine Learning model predicts whether a post is Fake, Real or Misleading.
""")
