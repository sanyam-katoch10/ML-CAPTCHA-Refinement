import streamlit as st
from io import BytesIO
from generator import generate_captcha
from refine_m import refine, predict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="ML CAPTCHA Refinement",
    page_icon="🔒",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: radial-gradient(circle at top left, #2e2e2e, #121212, #0b0b0b);
    background-size: 200% 200%;
    animation: chromeFlow 20s ease infinite;
    color: #e6e6e6;
}

@keyframes chromeFlow {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

/* ===== GLASS CHROME CARD ===== */
.chrome-card {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.06),
        rgba(255,255,255,0.015)
    );
    backdrop-filter: blur(20px);
    border-radius: 22px;
    padding: 26px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow:
        inset 0 0 12px rgba(255,255,255,0.05),
        0 20px 50px rgba(0,0,0,0.7);
}

/* ===== TITLES ===== */
.main-title {
    font-size: 54px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#f5f5f5,#9f9f9f,#f5f5f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #b5b5b5;
    margin-bottom: 45px;
}

/* ===== SHINY BUTTONS ===== */
.stButton button {
    width: 100%;
    border-radius: 16px;
    border: none;
    padding: 14px 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #ffffff;
    background: linear-gradient(
        135deg,
        #3a3a3a,
        #6f6f6f,
        #3a3a3a
    );
    box-shadow:
        inset 0 1px 1px rgba(255,255,255,0.4),
        0 6px 20px rgba(0,0,0,0.6);
    transition: all 0.35s ease;
}

.stButton button:hover {
    transform: translateY(-2px) scale(1.02);
    background: linear-gradient(
        135deg,
        #6f6f6f,
        #d1d1d1,
        #6f6f6f
    );
    box-shadow:
        0 0 20px rgba(200,200,200,0.35),
        0 10px 35px rgba(0,0,0,0.8);
}

/* ===== SLIDERS ===== */
.stSlider > div {
    color: #d6d6d6;
}

/* ===== FOOTER ===== */
.footer {
    text-align:center;
    color:#8a8a8a;
    margin-top:50px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🔒 ML CAPTCHA REFINEMENT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Adaptive CAPTCHA optimization using machine intelligence</div>', unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
col1, col2, col3 = st.columns([1.25, 1.8, 1.35])

# ---------- LEFT: MANUAL CONTROLS ----------
with col1:
    st.markdown('<div class="chrome-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Control Parameters")

    noise = st.slider("Noise Intensity", 0.0, 1.0, 0.25)
    distortion = st.slider("Geometric Distortion", 0.0, 1.0, 0.25)
    clutter = st.slider("Visual Clutter", 0.0, 1.0, 0.25)

    generate_btn = st.button("🎲 Generate CAPTCHA")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- CENTER: PREVIEW ----------
with col2:
    st.markdown('<div class="chrome-card">', unsafe_allow_html=True)
    st.markdown("### 🖼 CAPTCHA Output")

    if generate_btn:
        img, text = generate_captcha(noise, distortion, clutter)
        st.image(img, use_column_width=True)

        pred, conf = predict(img)
        st.markdown(
            f"""
            **Text:** `{text}`  
            **Predicted Difficulty:** `{pred.upper()}`  
            **Confidence Score:** `{conf:.2f}`
            """
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RIGHT: REFINEMENT ----------
with col3:
    st.markdown('<div class="chrome-card">', unsafe_allow_html=True)
    st.markdown("### 🔁 Refinement Engine")

    target = st.selectbox("Target Difficulty", ["easy", "medium", "hard"])
    refine_once = st.button("✨ Refine Once")
    auto_refine = st.button("🚀 Auto-Refine")

    line_slot = st.empty()
    heat_slot = st.empty()

    if refine_once:
        img, text, level = refine(target)
        st.image(img, use_column_width=True)

        buf = BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            "⬇ Download CAPTCHA",
            buf.getvalue(),
            file_name=f"{text}_{level}.png",
            mime="image/png"
        )

    if auto_refine:
        grid = 4
        convergence = []

        for step in range(6):
            matrix = np.zeros((grid, grid))

            for i in range(grid):
                for j in range(grid):
                    img, _, _ = refine(target)
                    _, c = predict(img)
                    matrix[i, j] = c

            convergence.append(matrix.mean())

            # Line plot
            fig1, ax1 = plt.subplots()
            ax1.plot(convergence, marker='o', linewidth=2)
            ax1.set_ylim(0, 1)
            ax1.set_title("Confidence Convergence")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Confidence")
            line_slot.pyplot(fig1, clear_figure=True)
            plt.close(fig1)

            # Heatmap
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            ax2.set_title(f"Difficulty Surface — Step {step+1}")
            heat_slot.pyplot(fig2, clear_figure=True)
            plt.close(fig2)

            time.sleep(0.45)

        st.success("Target difficulty stabilized ✔")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">✨ Designed & Engineered by SANYAM KATOCH ✨</div>', unsafe_allow_html=True)
