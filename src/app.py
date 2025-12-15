import streamlit as st
from io import BytesIO
from generator import generate_captcha
from refine_m import refine, predict


st.set_page_config(
    page_title="ML CAPTCHA Refinement",
    page_icon="🔐",
    layout="wide"
)


st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem;
}

/* Glass card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.15);
}

/* Title */
.hero-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
}

.hero-sub {
    text-align: center;
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 40px;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
}

.stButton button:hover {
    transform: scale(1.03);
}

/* Sliders */
.stSlider label {
    font-weight: 600;
    color: #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔐 ML CAPTCHA Refinement</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered CAPTCHA generation with adaptive difficulty optimization</div>',
    unsafe_allow_html=True
)


left, center, right = st.columns([1.1, 1.6, 1.1])


with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Manual Controls")

    noise = st.slider("Noise", 0.0, 1.0, 0.2)
    dist = st.slider("Distortion", 0.0, 1.0, 0.2)
    clutter = st.slider("Clutter", 0.0, 1.0, 0.2)

    gen_btn = st.button("🎲 Generate CAPTCHA", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with center:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🖼️ CAPTCHA Preview")

    if gen_btn:
        img, text = generate_captcha(noise, dist, clutter)
        st.image(img, use_column_width=True)
        pred, conf = predict(img)

        st.markdown(
            f"""
            **Text:** `{text}`  
            **Predicted Difficulty:** `{pred.upper()}`  
            **Confidence:** `{conf:.2f}`
            """
        )

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🎯 AI Refinement")

    target = st.selectbox(
        "Target Difficulty",
        ["easy", "medium", "hard"]
    )

    refine_btn = st.button("✨ Refine CAPTCHA", use_container_width=True)

    if refine_btn:
        img, text, predicted = refine(target)
        st.image(img, use_column_width=True)

        buf = BytesIO()
        img.save(buf, format="PNG")

        st.download_button(
            "⬇️ Download",
            data=buf.getvalue(),
            file_name=f"{text}_{predicted}.png",
            mime="image/png",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<center style='color:#9ca3af;margin-top:40px;'>Built with ❤️ using Streamlit & Deep Learning</center>",
    unsafe_allow_html=True
)
