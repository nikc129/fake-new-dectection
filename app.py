import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detection", layout="wide")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model_path = "model"  # folder containing pytorch_model.bin

    clf = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path
    )

    return clf


clf = load_model()


# -----------------------------
# Prediction Function
# -----------------------------
def predict_news(text):

    result = clf(text)[0]

    label = result["label"]
    score = result["score"]

    if label == "LABEL_0":
        return "FAKE NEWS", score
    else:
        return "REAL NEWS", score


# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection System")

st.write("Enter news content (max ~500 words) to check if it is real or fake.")

st.markdown("---")

news_text = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")

    else:

        with st.spinner("Analyzing news..."):

            label, confidence = predict_news(news_text)

        color = "green" if label == "REAL NEWS" else "red"

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:10px;
                text-align:center;
                color:white">
            <h2>{label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(f"Confidence Score: {confidence*100:.2f}%")

        st.progress(confidence)