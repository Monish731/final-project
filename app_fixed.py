import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Intelligent Customer Support",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(99,102,241,.16), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(16,185,129,.10), transparent 25%),
        #0b0d12;
}
.block-container {max-width:1450px;padding-top:2rem;padding-bottom:4rem;}
h1 {font-size:2.7rem!important;font-weight:800!important;letter-spacing:-.04em;}
h2,h3 {font-weight:750!important;}
.hero {
    padding:1.35rem 1.5rem;border:1px solid rgba(255,255,255,.08);
    border-radius:18px;
    background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(16,185,129,.06));
    margin-bottom:1.4rem;
}
.subtitle {color:#a7adbb;font-size:1.05rem;}
.section-title {font-size:1.45rem;font-weight:750;margin-top:1.5rem;margin-bottom:.8rem;}
.metric-card {
    min-height:145px;padding:1.15rem 1.2rem;border-radius:17px;
    border:1px solid rgba(255,255,255,.10);
    background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
    box-shadow:0 10px 30px rgba(0,0,0,.16);
}
.metric-label {color:#a7adbb;font-size:.88rem;font-weight:600;margin-bottom:.45rem;}
.metric-value {font-size:1.75rem;font-weight:800;color:#f4f6fb;line-height:1.15;}
.metric-sub {color:#8e96a5;font-size:.78rem;margin-top:.55rem;}
.intent-card {border-left:4px solid #818cf8;}
.priority-low {border-left:4px solid #22c55e;}
.priority-medium {border-left:4px solid #f59e0b;}
.priority-high {border-left:4px solid #ef4444;}
.response-box {
    padding:1.2rem 1.35rem;border-radius:16px;
    border:1px solid rgba(34,197,94,.24);
    background:linear-gradient(135deg,rgba(34,197,94,.16),rgba(16,185,129,.08));
    color:#e8fff0;font-size:1.02rem;line-height:1.65;
}
.method-badge {
    display:inline-block;padding:.35rem .75rem;border-radius:999px;
    font-size:.78rem;font-weight:750;margin-top:.7rem;margin-right:.4rem;
}
.badge-blue {background:rgba(99,102,241,.18);color:#c7d2fe;border:1px solid rgba(129,140,248,.28);}
.badge-green {background:rgba(34,197,94,.15);color:#bbf7d0;border:1px solid rgba(34,197,94,.25);}
.badge-orange {background:rgba(245,158,11,.15);color:#fde68a;border:1px solid rgba(245,158,11,.25);}
.summary-card {
    padding:1.2rem 1.35rem;border-radius:16px;
    border:1px solid rgba(255,255,255,.08);
    background:rgba(255,255,255,.025);line-height:1.9;
}
.summary-key {color:#9ca3af;font-weight:600;}
.footer-note {color:#7f8795;font-size:.82rem;text-align:center;margin-top:2rem;}
div[data-baseweb="textarea"] textarea,div[data-baseweb="input"] input {border-radius:12px!important;}
.stButton>button {min-height:3.1rem!important;border-radius:12px!important;font-weight:750!important;}
.stButton>button:hover {transform:translateY(-1px);box-shadow:0 8px 22px rgba(99,102,241,.18);}
div[data-testid="stDataFrame"] {border-radius:14px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent

MODEL_FILES = [
    "ps1_intent_svm_model.pkl",
    "ps1_tfidf_vectorizer.pkl",
    "ps2_priority_svm_model.pkl",
    "ps2_priority_tfidf_vectorizer.pkl",
    "ps3_encoder.keras",
    "ps3_decoder.keras",
    "ps3_query_tokenizer.pkl",
    "ps3_response_tokenizer.pkl",
    "ps3_retrieval_bundle.pkl",
    "ps4_xgboost_bundle.pkl",
]

@st.cache_resource
def load_models():
    missing = [x for x in MODEL_FILES if not (BASE / x).exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n- " + "\n- ".join(missing))

    ps1_model = joblib.load(BASE / "ps1_intent_svm_model.pkl")
    ps1_vectorizer = joblib.load(BASE / "ps1_tfidf_vectorizer.pkl")

    ps2_model = joblib.load(BASE / "ps2_priority_svm_model.pkl")
    ps2_vectorizer = joblib.load(BASE / "ps2_priority_tfidf_vectorizer.pkl")

    encoder = tf.keras.models.load_model(BASE / "ps3_encoder.keras", compile=False)
    decoder = tf.keras.models.load_model(BASE / "ps3_decoder.keras", compile=False)
    query_tokenizer = joblib.load(BASE / "ps3_query_tokenizer.pkl")
    response_tokenizer = joblib.load(BASE / "ps3_response_tokenizer.pkl")

    retrieval_bundle = joblib.load(BASE / "ps3_retrieval_bundle.pkl")
    retrieval_vectorizer = retrieval_bundle["vectorizer"]
    retrieval_matrix = retrieval_bundle["matrix"]
    retrieval_responses = retrieval_bundle["responses"]

    ps4_bundle = joblib.load(BASE / "ps4_xgboost_bundle.pkl")
    ps4_model = ps4_bundle["model"]
    ps4_encoder = ps4_bundle["categorical_encoder"]
    ps4_categorical_features = ps4_bundle.get(
        "categorical_features", ["intent", "priority", "company"]
    )
    ps4_numeric_features = ps4_bundle.get(
        "numeric_features",
        ["message_length", "word_count", "hour", "day_of_week", "urgency"]
    )

    return {
        "ps1_model": ps1_model, "ps1_vectorizer": ps1_vectorizer,
        "ps2_model": ps2_model, "ps2_vectorizer": ps2_vectorizer,
        "encoder": encoder, "decoder": decoder,
        "query_tokenizer": query_tokenizer,
        "response_tokenizer": response_tokenizer,
        "retrieval_vectorizer": retrieval_vectorizer,
        "retrieval_matrix": retrieval_matrix,
        "retrieval_responses": retrieval_responses,
        "ps4_model": ps4_model, "ps4_encoder": ps4_encoder,
        "ps4_categorical_features": ps4_categorical_features,
        "ps4_numeric_features": ps4_numeric_features,
    }

try:
    models = load_models()
except Exception as exc:
    st.error("❌ Model loading failed.")
    st.code(str(exc))
    st.stop()

ps1_model = models["ps1_model"]
ps1_vectorizer = models["ps1_vectorizer"]
ps2_model = models["ps2_model"]
ps2_vectorizer = models["ps2_vectorizer"]
encoder = models["encoder"]
decoder = models["decoder"]
query_tokenizer = models["query_tokenizer"]
response_tokenizer = models["response_tokenizer"]
retrieval_vectorizer = models["retrieval_vectorizer"]
retrieval_matrix = models["retrieval_matrix"]
retrieval_responses = models["retrieval_responses"]
ps4_model = models["ps4_model"]
ps4_encoder = models["ps4_encoder"]
ps4_categorical_features = models["ps4_categorical_features"]

def clean_query(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_retrieved_response(response):
    if not response:
        return ""
    response = str(response).strip()
    response = re.sub(r"http\S+|www\.\S+", "", response, flags=re.I)
    response = re.sub(r"@\w+", "", response)
    response = re.sub(r"\s+", " ", response).strip()

    keep_two_letter = {
        "am","an","as","at","be","by","do","go","he","if","in","is",
        "it","me","my","no","of","on","or","so","to","up","us","we"
    }
    parts = response.split()
    while parts:
        last = re.sub(r"[^a-z]", "", parts[-1].lower())
        if (len(last) <= 2 and last and last not in keep_two_letter
                and not parts[-1].endswith((".", "!", "?", ","))):
            parts.pop()
        else:
            break
    return " ".join(parts).strip()

def is_poor_response(query, response):
    if not response:
        return True
    response = str(response).strip().lower()
    words = response.split()

    if len(words) < 5 or len(words) > 50:
        return True
    if any(x in response for x in ["<unk>", "<start>", "<end>"]):
        return True

    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            return True

    counts = {}
    for word in words:
        word = re.sub(r"[^a-z]", "", word)
        if word:
            counts[word] = counts.get(word, 0) + 1
    if counts and max(counts.values()) >= 5:
        return True

    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    if any(bigrams.count(x) >= 3 for x in set(bigrams)):
        return True

    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if any(trigrams.count(x) >= 2 for x in set(trigrams)):
        return True

    if len(words) >= 10 and len(set(words)) / len(words) < 0.45:
        return True

    garbage = [
        "get you to get", "help you to get",
        "to get this to", "this to get you",
        "sorry sorry", "help help", "please please"
    ]
    if any(x in response for x in garbage):
        return True

    if response.count(" sorry ") > 2:
        return True

    stop = {
        "the","a","an","is","are","am","to","of","for","and","or",
        "i","my","me","we","you","your","it","this","that","with",
        "on","in"
    }
    qwords = set(re.findall(r"[a-z]+", clean_query(query))) - stop
    rwords = set(re.findall(r"[a-z]+", response))
    if len(qwords) >= 3 and not (qwords & rwords):
        return True

    return False

def generate_seq2seq_response(query):
    sequence = query_tokenizer.texts_to_sequences([clean_query(query)])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=50, padding="post", truncating="post"
    )

    encoder_result = encoder.predict(sequence, verbose=0)
    if not isinstance(encoder_result, (list, tuple)):
        raise ValueError("Unexpected PS-3 encoder output.")

    if len(encoder_result) == 3:
        enc_outputs, state_h, state_c = encoder_result
    elif len(encoder_result) == 2:
        enc_outputs = None
        state_h, state_c = encoder_result
    else:
        raise ValueError("Unsupported PS-3 encoder output format.")

    start_id = response_tokenizer.word_index.get("<start>", 2)
    end_id = response_tokenizer.word_index.get("<end>", 3)
    target_seq = np.array([[start_id]], dtype=np.int32)
    generated_words = []

    for _ in range(59):
        if enc_outputs is not None:
            decoder_result = decoder.predict(
                [target_seq, enc_outputs, state_h, state_c], verbose=0
            )
        else:
            decoder_result = decoder.predict(
                [target_seq, state_h, state_c], verbose=0
            )

        if isinstance(decoder_result, (list, tuple)):
            predictions = decoder_result[0]
            if len(decoder_result) >= 3:
                state_h = decoder_result[-2]
                state_c = decoder_result[-1]
        else:
            predictions = decoder_result

        predictions = np.asarray(predictions)
        if predictions.ndim == 3:
            probs = predictions[0, -1, :]
        elif predictions.ndim == 2:
            probs = predictions[-1, :]
        else:
            raise ValueError(f"Unexpected decoder shape: {predictions.shape}")

        token_id = int(np.argmax(probs))
        if token_id == end_id:
            break

        word = response_tokenizer.index_word.get(token_id, "")
        if word not in {"", "<start>", "<end>", "<unk>"}:
            generated_words.append(word)

        target_seq = np.array([[token_id]], dtype=np.int32)

    return " ".join(generated_words).strip()

def retrieve_response(query):
    """Retrieve a response only when the match is genuinely relevant."""
    cleaned_query = clean_query(query)

    query_vector = retrieval_vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(
        query_vector,
        retrieval_matrix
    )[0]

    candidate_indices = np.argsort(similarities)[::-1][:10]

    query_words = set(re.findall(r"[a-z]+", cleaned_query))
    stop_words = {
        "the", "a", "an", "is", "are", "am", "to", "of", "for",
        "and", "or", "i", "my", "me", "we", "you", "your", "it",
        "this", "that", "with", "on", "in", "was", "were", "has",
        "have", "had"
    }
    useful_query_words = query_words - stop_words

    best_index = None
    best_adjusted_score = -1.0

    for idx in candidate_indices:
        score = float(similarities[idx])
        response = clean_retrieved_response(
            retrieval_responses[idx]
        )

        if not response:
            continue

        response_words = set(
            re.findall(r"[a-z]+", response.lower())
        )
        overlap = useful_query_words & response_words

        # Small relevance bonus for shared meaningful words.
        overlap_bonus = min(len(overlap) * 0.05, 0.20)
        adjusted_score = score + overlap_bonus

        if adjusted_score > best_adjusted_score:
            best_adjusted_score = adjusted_score
            best_index = int(idx)

    if best_index is None:
        return "", 0.0

    original_score = float(similarities[best_index])
    response = clean_retrieved_response(
        retrieval_responses[best_index]
    )

    # Reject weak matches. This prevents unrelated support replies.
    if original_score < 0.35:
        return "", original_score

    # Reject repetitive / malformed retrieved responses too.
    if is_poor_response(query, response):
        return "", original_score

    return response, original_score


def contextual_fallback(query):
    """Return a category-specific PS-3 response."""

    text = clean_query(query)

    # --------------------------------------------------------
    # BILLING / PAYMENT / REFUND
    # IMPORTANT: check this BEFORE delivery/order because
    # billing queries can also contain the word "order".
    # --------------------------------------------------------
    if any(x in text for x in [
        "charged twice",
        "double charged",
        "duplicate charge",
        "wrong charge",
        "billing",
        "payment",
        "paid twice",
        "payment failed",
        "payment pending",
        "transaction",
        "refund",
        "money back",
        "reimbursement",
        "charged",
        "debited",
        "amount deducted",
        "money deducted",
    ]):
        if any(x in text for x in [
            "refund",
            "money back",
            "reimbursement",
        ]):
            return (
                "We're sorry for the delay with your refund. "
                "Please send us your order or transaction details "
                "so our support team can check the refund status "
                "and assist you."
            )

        if any(x in text for x in [
            "charged twice",
            "double charged",
            "duplicate charge",
            "paid twice",
        ]):
            return (
                "We're sorry about the duplicate payment. "
                "Please send us your transaction or order details "
                "so our support team can investigate the duplicate "
                "charge and assist you."
            )

        if any(x in text for x in [
            "pending",
            "payment pending",
            "paid",
        ]):
            return (
                "We're sorry about the payment issue. "
                "Please send us your transaction or order details "
                "so our support team can check the payment status "
                "and assist you."
            )

        return (
            "We're sorry about the payment issue. "
            "Please send us your transaction or order details "
            "so our support team can investigate and assist you."
        )

    # --------------------------------------------------------
    # DELIVERY / PACKAGE
    # Do NOT trigger this merely because the query contains
    # "order". It must contain a delivery/shipping concept.
    # --------------------------------------------------------
    if any(x in text for x in [
        "package",
        "parcel",
        "shipment",
        "shipping",
        "delivery",
        "tracking",
        "courier",
        "arrive",
        "arrived",
        "dispatched",
        "dispatch",
    ]):
        if any(x in text for x in [
            "not",
            "hasn't",
            "hasnt",
            "late",
            "delayed",
            "missing",
            "where",
            "still",
            "yet",
            "pending",
            "not received",
        ]):
            return (
                "We're sorry your package has not arrived yet. "
                "Please send us your order or tracking details so "
                "our support team can check the shipment status "
                "and assist you."
            )

        return (
            "We'd be happy to help with your delivery. "
            "Please send us your order or tracking details so "
            "our support team can check the latest shipment status."
        )

    # --------------------------------------------------------
    # ACCOUNT / LOGIN
    # --------------------------------------------------------
    if any(x in text for x in [
        "cannot login",
        "can't login",
        "cannot log in",
        "can't log in",
        "unable to login",
        "unable to log in",
        "password",
        "account locked",
        "account access",
        "login",
        "log in",
    ]):
        return (
            "We're sorry you're having trouble accessing your "
            "account. Please send us more details so our support "
            "team can help you regain access."
        )

    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------
    if any(x in text for x in [
        "not working",
        "doesn't work",
        "doesnt work",
        "error",
        "crash",
        "crashing",
        "technical issue",
        "technical problem",
        "bug",
        "website is down",
        "application is down",
        "app is down",
    ]):
        return (
            "We're sorry you're experiencing this technical issue. "
            "Please send us the details so our support team can "
            "investigate and help resolve it."
        )

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------
    if any(x in text for x in [
        "cancel my order",
        "cancel the order",
        "cancelled my order",
        "canceled my order",
        "order was cancelled",
        "order was canceled",
        "cancelled without my permission",
        "canceled without my permission",
    ]):
        return (
            "We're sorry about the order cancellation. "
            "Please send us your order details so our support "
            "team can check what happened and assist you."
        )

    # --------------------------------------------------------
    # PRODUCT INFORMATION
    # --------------------------------------------------------
    if any(x in text for x in [
        "product",
        "product information",
        "product details",
        "tell me more about this product",
        "features",
        "specifications",
    ]):
        return (
            "We'd be happy to provide more information about "
            "the product. Please share the product name or the "
            "details you're looking for, and our support team "
            "will assist you."
        )

    # --------------------------------------------------------
    # ORDER INFORMATION
    # --------------------------------------------------------
    if any(x in text for x in [
        "order status",
        "track my order",
        "where is my order",
        "order details",
    ]):
        return (
            "We'd be happy to help with your order. "
            "Please send us your order number so our support "
            "team can check the latest status."
        )

    # --------------------------------------------------------
    # SUPPORT HOURS
    # --------------------------------------------------------
    if any(x in text for x in [
        "support hours",
        "customer support hours",
        "working hours",
        "opening hours",
        "business hours",
    ]):
        return (
            "Our support team is available to help with your "
            "query. Please contact us through the available "
            "support channel for the current support hours."
        )

    # --------------------------------------------------------
    # COMPLAINT
    # --------------------------------------------------------
    if any(x in text for x in [
        "complaint",
        "terrible",
        "worst",
        "unacceptable",
        "disappointed",
        "poor service",
        "bad service",
    ]):
        return (
            "We're sorry to hear about your experience. "
            "Please send us the details so our support team "
            "can review the issue and assist you."
        )

    # --------------------------------------------------------
    # GENERAL INQUIRY
    # --------------------------------------------------------
    return (
        "Thanks for contacting us. Please share a few more "
        "details about the issue so our support team can assist you."
    )


def get_final_response(query):
    """
    PS-3 decision flow:

    Seq2Seq LSTM + Attention
             |
        quality check
             |
       +-----+-----+
       |           |
     good         poor
       |           |
    return      TF-IDF retrieval
                   |
              similarity >= 0.35
                   |
              +----+----+
              |         |
            good       weak
              |         |
           return   contextual fallback
    """

    try:
        generated = generate_seq2seq_response(query)
    except Exception:
        generated = ""

    if generated and not is_poor_response(query, generated):
        return {
            "response": generated,
            "method": "Seq2Seq LSTM + Attention",
            "similarity": None,
        }

    try:
        retrieved, similarity = retrieve_response(query)

        if retrieved:
            return {
                "response": retrieved,
                "method": "TF-IDF Retrieval",
                "similarity": similarity,
            }
    except Exception:
        pass

    return {
        "response": contextual_fallback(query),
        "method": "Contextual Fallback",
        "similarity": None,
    }


def create_ps4_features(query, company, intent, priority):
    text = clean_query(query)
    urgency_terms = [
        "urgent","urgently","immediately","asap","emergency","critical",
        "worst","cancel","cancelled","canceled","not working","charged twice",
        "cannot login","can't login","not received","missing","late","delayed","now"
    ]
    urgency = int(any(x in text for x in urgency_terms))

    numeric = np.array([[
        len(query), len(query.split()),
        datetime.now().hour, datetime.now().weekday(), urgency
    ]], dtype=float)

    categorical = pd.DataFrame([[
        str(intent).strip() if intent else "General Inquiry",
        str(priority).strip() if priority else "Low",
        str(company).strip() if company and str(company).strip() else "unknown",
    ]], columns=ps4_categorical_features)

    encoded = ps4_encoder.transform(categorical)
    return sp.hstack([sp.csr_matrix(numeric), encoded]).tocsr()

def format_response_time(minutes):
    minutes = float(minutes)
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hr"
    return f"{hours/24:.1f} days"

def priority_class(priority):
    value = str(priority).lower()
    if "high" in value:
        return "priority-high"
    if "medium" in value:
        return "priority-medium"
    return "priority-low"


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>💬 Intelligent Customer Support</h1>
    <div class="subtitle">
        AI-powered ticket analysis across intent, priority,
        response generation, and expected response time.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 AI Pipeline")
    st.markdown("""
**PS-1** — Intent Classification  
`LinearSVC`

**PS-2** — Priority Detection  
`LinearSVC`

**PS-3** — Response Generation  
`Seq2Seq LSTM + Attention`  
`TF-IDF fallback`

**PS-4** — Response-Time Prediction  
`XGBoost`
""")
    st.divider()
    st.markdown("### 📌 Processing")
    st.caption(
        "Understand → Prioritize → Generate → "
        "Quality-check → Retrieve if needed → Predict time"
    )

st.markdown(
    '<div class="section-title">🎫 Customer Ticket</div>',
    unsafe_allow_html=True,
)

query = st.text_area(
    "Enter customer query",
    placeholder="Example: My package has not arrived yet",
    height=130,
)

company = st.text_input(
    "Support Company / Handle",
    placeholder="Example: AmazonHelp",
)

analyze = st.button(
    "🚀 Analyze Ticket",
    type="primary",
    use_container_width=True,
)

if analyze:
    if not query.strip():
        st.warning("Please enter a customer query.")
        st.stop()

    with st.spinner("Analyzing ticket..."):
        X1 = ps1_vectorizer.transform([query])
        intent = ps1_model.predict(X1)[0]

        X2 = ps2_vectorizer.transform([query])
        priority = ps2_model.predict(X2)[0]

        ps3_result = get_final_response(query)

        response = ps3_result["response"]
        response_method = ps3_result["method"]
        response_similarity = ps3_result["similarity"]

        ps4_error = None
        predicted_time = 0.0

        try:
            features = create_ps4_features(
                query, company, intent, priority
            )
            predicted_time = float(ps4_model.predict(features)[0])
            predicted_time = max(0.0, predicted_time)
        except Exception as exc:
            ps4_error = str(exc)

    st.divider()
    st.markdown(
        '<div class="section-title">📊 Ticket Analysis</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
<div class="metric-card intent-card">
<div class="metric-label">Predicted Intent</div>
<div class="metric-value">{intent}</div>
<div class="metric-sub">PS-1 • LinearSVC</div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="metric-card {priority_class(priority)}">
<div class="metric-label">Priority</div>
<div class="metric-value">{priority}</div>
<div class="metric-sub">PS-2 • LinearSVC</div>
</div>
""", unsafe_allow_html=True)

    with c3:
        time_display = (
            format_response_time(predicted_time)
            if ps4_error is None else "Unavailable"
        )
        time_sub = (
            f"PS-4 • XGBoost • {predicted_time:.1f} minutes"
            if ps4_error is None else "PS-4 prediction error"
        )
        st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Estimated Response Time</div>
<div class="metric-value">{time_display}</div>
<div class="metric-sub">{time_sub}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<span class="method-badge badge-blue">PS-1 • LinearSVC</span>
<span class="method-badge badge-green">PS-2 • Priority</span>
<span class="method-badge badge-blue">PS-4 • XGBoost</span>
""", unsafe_allow_html=True)

    if ps4_error is None:
        st.caption(f"Model prediction: {predicted_time:.1f} minutes")

    st.markdown(
        '<div class="section-title">🤖 Suggested Company Response</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="response-box">{response}</div>',
        unsafe_allow_html=True,
    )

    if response_method == "Seq2Seq LSTM + Attention":
        st.markdown(
            '<span class="method-badge badge-blue">'
            'PS-3 • Seq2Seq LSTM + Attention'
            '</span>',
            unsafe_allow_html=True,
        )
        st.caption("Generated response passed the PS-3 quality checks.")

    elif response_method == "TF-IDF Retrieval":
        st.markdown(
            '<span class="method-badge badge-green">'
            'PS-3 • TF-IDF Retrieval Fallback'
            '</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Seq2Seq output was rejected as low quality. "
            f"TF-IDF retrieved a relevant support response "
            f"(similarity: {response_similarity:.3f})."
        )

    else:
        st.markdown(
            '<span class="method-badge badge-orange">'
            'PS-3 • Contextual Fallback'
            '</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Generation and retrieval were not sufficiently usable; "
            "a contextual fallback was used."
        )

    st.divider()
    st.markdown(
        '<div class="section-title">📝 Ticket Summary</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown(f"""
<div class="summary-card">
<div><span class="summary-key">Customer Query:</span> {query}</div>
<div><span class="summary-key">Support Company:</span>
{company if company else "Not specified"}</div>
<div><span class="summary-key">Intent:</span> {intent}</div>
</div>
""", unsafe_allow_html=True)

    with right:
        time_summary = (
            f"{format_response_time(predicted_time)} "
            f"({predicted_time:.1f} minutes)"
            if ps4_error is None else "Unavailable"
        )
        st.markdown(f"""
<div class="summary-card">
<div><span class="summary-key">Priority:</span> {priority}</div>
<div><span class="summary-key">Estimated Response Time:</span> {time_summary}</div>
<div><span class="summary-key">PS-3 Method:</span> {response_method}</div>
</div>
""", unsafe_allow_html=True)

    if ps4_error is not None:
        st.warning("⚠️ PS-4 response-time prediction could not be generated.")
        with st.expander("Show PS-4 technical details"):
            st.code(ps4_error)
    elif predicted_time > 1440:
        st.error("🚨 Potential SLA breach: predicted response time exceeds 24 hours.")
    elif predicted_time > 480:
        st.warning("⚠️ Potential SLA risk: predicted response time exceeds 8 hours.")
    else:
        st.success("✅ Predicted response time is within normal range.")

st.divider()
st.markdown(
    '<div class="section-title">🔬 Models Used</div>',
    unsafe_allow_html=True,
)

models_df = pd.DataFrame({
    "Problem Statement": ["PS-1", "PS-2", "PS-3", "PS-4"],
    "Task": [
        "Intent Classification",
        "Priority Detection",
        "Response Generation",
        "Response-Time Prediction",
    ],
    "Model": [
        "LinearSVC",
        "LinearSVC",
        "Seq2Seq LSTM + Attention + TF-IDF Fallback",
        "XGBoost Regressor",
    ],
})

st.dataframe(
    models_df,
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    '<div class="footer-note">'
    'Intelligent Customer Support Ticket System • PS-1 + PS-2 + PS-3 + PS-4'
    '</div>',
    unsafe_allow_html=True,
)
