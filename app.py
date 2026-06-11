import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from PIL import Image
import easyocr

model = joblib.load("logistic_regression_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

@st.cache_resource
def load_ocr_reader():
    """
    Load EasyOCR only when the user explicitly requests OCR.

    download_enabled=False prevents Streamlit Cloud from trying to download
    large OCR model files at runtime, which can crash the app process.
    If the OCR model is not already available in the deployment environment,
    the app will show a graceful warning instead of crashing.
    """
    return easyocr.Reader(["en"], gpu=False, download_enabled=False, verbose=False)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_risk_level(fake_prob):
    if fake_prob >= 0.70:
        return "High Risk", "This review is highly suspicious."
    elif fake_prob >= 0.40:
        return "Medium Risk", "This review should be interpreted carefully."
    else:
        return "Low Risk", "This review appears relatively reliable."

def get_recommendation(fake_prob):
    if fake_prob >= 0.70:
        return "Do Not Buy Based on This Review"
    elif fake_prob >= 0.40:
        return "Be Careful Before Buying"
    else:
        return "Reasonably Safe to Consider"

def analyze_quality(text):
    words = text.split()
    word_count = len(words)

    promotional_words = [
        "amazing", "perfect", "best", "excellent", "awesome",
        "highly recommend", "must buy", "life changing", "incredible"
    ]

    specific_words = [
        "battery", "shipping", "size", "quality", "price",
        "screen", "material", "delivery", "packaging", "fit"
    ]

    negative_words = [
        "but", "however", "although", "problem", "issue",
        "expensive", "slow", "weak", "disappointed"
    ]

    lower_text = text.lower()

    promo_count = sum(1 for w in promotional_words if w in lower_text)
    specific_count = sum(1 for w in specific_words if w in lower_text)
    negative_count = sum(1 for w in negative_words if w in lower_text)

    if word_count < 20:
        length_eval = "Short"
    elif word_count < 60:
        length_eval = "Moderate"
    else:
        length_eval = "Long"

    tone_eval = "Highly promotional" if promo_count >= 2 else "Moderately promotional" if promo_count == 1 else "Neutral"

    specificity_eval = "High" if specific_count >= 2 else "Moderate" if specific_count == 1 else "Low"

    balance_eval = "Balanced" if negative_count >= 1 else "One-sided"

    return word_count, length_eval, tone_eval, specificity_eval, balance_eval

def get_influential_words(review, top_n=8):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    coef = model.coef_[0]
    nonzero_indices = vec.nonzero()[1]

    contributions = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        score = vec[0, idx] * coef[idx]
        contributions.append((word, score))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    fake_words = [w for w, s in contributions if s > 0][:top_n]
    real_words = [w for w, s in contributions if s < 0][:top_n]

    return fake_words, real_words

def get_word_importance_dataframe(review, top_n=10):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    coef = model.coef_[0]
    nonzero_indices = vec.nonzero()[1]

    contributions = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        score = float(vec[0, idx] * coef[idx])
        direction = "Fake" if score > 0 else "Real"
        contributions.append((word, score, direction))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]

    if not contributions:
        return pd.DataFrame(columns=["Word or Phrase", "Influence Score", "Direction"])

    importance_df = pd.DataFrame(
        contributions,
        columns=["Word or Phrase", "Influence Score", "Direction"]
    )

    return importance_df

def extract_text_from_image(uploaded_image):
    """
    Try to extract text from an uploaded review screenshot.

    Returns:
        extracted_text (str): OCR output if successful.
        error_message (str or None): Error message if OCR fails.

    This function is intentionally wrapped in try/except so that OCR failure
    does not break the rest of the Streamlit app.
    """
    try:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        reader = load_ocr_reader()
        results = reader.readtext(image_np, detail=0)

        extracted_text = " ".join(results).strip()

        if extracted_text == "":
            return "", "No readable text was detected in the image."

        return extracted_text, None

    except Exception as e:
        return "", str(e)

def analyze_and_display(review):
    cleaned_review = clean_text(review)
    review_vector = vectorizer.transform([cleaned_review])

    prediction = model.predict(review_vector)[0]
    probabilities = model.predict_proba(review_vector)[0]

    real_prob = probabilities[0]
    fake_prob = probabilities[1]

    risk_level, risk_description = get_risk_level(fake_prob)
    recommendation = get_recommendation(fake_prob)

    st.divider()

    st.header("1. Prediction Result")

    if prediction == 1:
        st.error("Prediction: Fake Review")
    else:
        st.success("Prediction: Real Review")

    st.header("2. Risk Level")
    st.subheader(risk_level)
    st.write(risk_description)

    st.header("3. Probability Score")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Fake Review Probability", f"{fake_prob * 100:.2f}%")

    with col4:
        st.metric("Real Review Probability", f"{real_prob * 100:.2f}%")

    st.progress(float(fake_prob))

    st.header("4. Natural Language Explanation")

    if fake_prob >= 0.70:
        st.write(
            f"The model predicts this review as fake with a probability of {fake_prob * 100:.2f}%. "
            "This suggests that the language pattern is strongly similar to fake reviews in the training dataset."
        )
    elif fake_prob >= 0.40:
        st.write(
            f"The model predicts this review as fake with a probability of {fake_prob * 100:.2f}%. "
            "Because the probability is moderate, this should be interpreted as a caution signal rather than definite proof."
        )
    else:
        st.write(
            f"The model predicts this review as relatively reliable, with only {fake_prob * 100:.2f}% probability of being fake. "
            "However, this result should still be combined with other information before making a purchase decision."
        )

    st.header("5. Why This Prediction?")

    fake_words, real_words = get_influential_words(review)

    if prediction == 1:
        st.write(
            "The model classified this review as fake because some words or phrases in the review are statistically associated "
            "with fake reviews in the training data."
        )
    else:
        st.write(
            "The model classified this review as real because some words or phrases in the review are statistically associated "
            "with original reviews in the training data."
        )

    st.header("6. Influential Words")

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Words pushing toward Fake")
        if fake_words:
            for word in fake_words:
                st.write(f"- {word}")
        else:
            st.write("No strong fake-related words detected.")

    with col6:
        st.subheader("Words pushing toward Real")
        if real_words:
            for word in real_words:
                st.write(f"- {word}")
        else:
            st.write("No strong real-related words detected.")

    st.header("7. Word Importance Visualization")

    importance_df = get_word_importance_dataframe(review)

    if importance_df.empty:
        st.write("No meaningful word-level influence scores were detected for this review.")
    else:
        st.write(
            "The chart below shows which words or phrases had the strongest influence on the prediction. "
            "Positive scores push the prediction toward Fake Review, while negative scores push it toward Real Review."
        )

        chart_df = importance_df[["Word or Phrase", "Influence Score"]].set_index("Word or Phrase")
        st.bar_chart(chart_df)

        with st.expander("View detailed word influence table"):
            st.dataframe(importance_df, width="stretch")

    st.header("8. Review Quality Analysis")

    word_count, length_eval, tone_eval, specificity_eval, balance_eval = analyze_quality(review)

    col7, col8, col9, col10 = st.columns(4)

    with col7:
        st.metric("Word Count", word_count)

    with col8:
        st.metric("Review Length", length_eval)

    with col9:
        st.metric("Specificity", specificity_eval)

    with col10:
        st.metric("Balance", balance_eval)

    st.write(f"Tone: **{tone_eval}**")

    st.header("9. Purchase Decision Checklist")

    st.write("""
    Before making a purchase decision, check the following:

    - Does the review mention specific product details?
    - Are there both positive and negative points?
    - Are many reviews written in a very similar style?
    - Is the tone overly promotional?
    - Does the product have consistent ratings across multiple reviews?
    """)

    st.header("10. Final Purchase Recommendation")
    st.subheader(recommendation)

    if fake_prob >= 0.70:
        st.write(
            "This review should not be used as the main basis for buying the product. "
            "Look for verified reviews and compare multiple sources before making a decision."
        )
    elif fake_prob >= 0.40:
        st.write(
            "This review may still contain useful information, but it should be interpreted carefully. "
            "It is better to compare it with other reviews before buying."
        )
    else:
        st.write(
            "This review appears relatively reliable, but purchase decisions should still consider product price, seller reputation, and other reviews."
        )

    with st.expander("11. Model Limitations"):
        st.write(
            "This model only analyzes review text. It does not consider reviewer history, verified purchase status, "
            "rating patterns, product images, seller information, or whether multiple reviews were posted from the same account. "
            "If an image is uploaded, OCR errors may occur when the screenshot is blurry, low-resolution, cropped, or contains non-English text. "
            "Therefore, the prediction should be used as a decision-support tool rather than absolute proof."
        )

st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Fake Review Detection System")

st.write(
    "This application uses TF-IDF, unigram/bigram n-gram features, and Logistic Regression "
    "to predict whether a product review is real or fake. Users can type a review directly "
    "or upload a screenshot containing review text. The system then provides an explanation "
    "and a purchase decision recommendation."
)

st.sidebar.header("Model Performance")
st.sidebar.write("Accuracy: 89.54%")
st.sidebar.write("Real Review F1-score: 0.90")
st.sidebar.write("Fake Review F1-score: 0.89")
st.sidebar.write("Macro Avg F1-score: 0.90")

st.sidebar.header("Model Design")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("N-gram range: unigram + bigram")
st.sidebar.write("Classifier: Logistic Regression")
st.sidebar.write("OCR: EasyOCR")

st.subheader("Choose Input Method")
input_method = st.radio(
    "Select how you want to provide the review:",
    ["Type Review Text", "Upload Review Screenshot"],
    horizontal=True
)

review = ""

if input_method == "Type Review Text":
    st.subheader("Enter a Review")

    review = st.text_area(
        "Enter a product review:",
        height=180,
        placeholder="Paste or type a product review here."
    )

else:
    st.subheader("Upload a Review Screenshot")

    uploaded_image = st.file_uploader(
        "Upload an image containing a product review:",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Review Screenshot", width="stretch")

        st.info(
            "OCR is an optional feature. If text extraction fails in the cloud environment, "
            "please type or paste the review text manually in the box below."
        )

        if st.button("Extract Text from Screenshot"):
            with st.spinner("Trying to extract text from image..."):
                extracted_text, ocr_error = extract_text_from_image(uploaded_image)

            if ocr_error:
                st.error("OCR could not process this image in the current deployment environment.")
                st.warning("Please type the review manually below.")
                with st.expander("Technical OCR error"):
                    st.write(ocr_error)
                st.session_state["ocr_review_text"] = ""
            else:
                st.success("Text extraction completed.")
                st.session_state["ocr_review_text"] = extracted_text

        if "ocr_review_text" not in st.session_state:
            st.session_state["ocr_review_text"] = ""

        st.subheader("Review Text for Analysis")
        review = st.text_area(
            "If OCR fails, type or paste the review text here:",
            value=st.session_state["ocr_review_text"],
            height=180,
            placeholder="Paste or type the product review here."
        )

st.subheader("How to Interpret the Result")
st.write("""
- Fake Probability ≥ 70%: High risk of fake review
- Fake Probability between 40% and 70%: Be cautious
- Fake Probability < 40%: Relatively reliable review
""")

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review first or upload a readable review screenshot.")
    else:
        analyze_and_display(review)
