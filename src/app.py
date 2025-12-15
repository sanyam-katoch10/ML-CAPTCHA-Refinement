import streamlit as st
from io import BytesIO
from PIL import Image
from generator import generate_captcha
from refine_m import refine, predict

st.set_page_config(
    page_title="ML CAPTCHA Refinement",
    page_icon="🔐",
    layout="wide"
)


st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #6c757d;
    margin-bottom: 30px;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background-color: #0e1117;
    border: 1px solid #262730;
}
.metric {
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔐 ML CAPTCHA Refinement System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-driven CAPTCHA generation & difficulty optimization</div>',
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ CAPTCHA Controls")

st.sidebar.subheader("Manual Parameters")
noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2)
dist = st.sidebar.slider("Distortion Level", 0.0, 1.0, 0.2)
clutter = st.sidebar.slider("Clutter Level", 0.0, 1.0, 0.2)

st.sidebar.divider()

st.sidebar.subheader("🎯 Target Difficulty")
target = st.sidebar.selectbox(
    "Refinement Goal",
    ["easy", "medium", "hard"]
)


col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🖼️ CAPTCHA Preview")
    preview_box = st.container()

with col2:
    st.markdown("### 📊 Model Feedback")
    feedback_box = st.container()


if st.sidebar.button("🎲 Generate CAPTCHA", use_container_width=True):
    img, text = generate_captcha(noise, dist, clutter)

    with preview_box:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.caption(f"Ground Truth Text: **{text}**")
        st.markdown('</div>', unsafe_allow_html=True)

    pred_label, conf = predict(img)

    with feedback_box:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success("Prediction Complete ✅")
        st.markdown(f"**Predicted Difficulty:** `{pred_label.upper()}`")
        st.markdown(f"**Confidence:** `{conf:.2f}`")
        st.markdown('</div>', unsafe_allow_html=True)


st.divider()
st.markdown("## 🔁 Target Difficulty Refinement")

if st.button("✨ Refine CAPTCHA", use_container_width=True):
    img, text, predicted = refine(target)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img, use_column_width=True)
    st.markdown(
        f"**Text:** `{text}` &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Predicted Difficulty:** `{predicted.upper()}`"
    )

    buf = BytesIO()
    img.save(buf, format="PNG")

    st.download_button(
        "⬇️ Download CAPTCHA",
        data=buf.getvalue(),
        file_name=f"{text}_{predicted}.png",
        mime="image/png",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    "<center>🚀 Built with Streamlit & Deep Learning</center>",
    unsafe_allow_html=True
)
