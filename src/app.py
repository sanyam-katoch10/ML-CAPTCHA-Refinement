import streamlit as st
from io import BytesIO
from generator import generate_captcha
from refine_m import refine, predict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

st.set_page_config(page_title="ML CAPTCHA Live Dashboard", layout="wide")

# ===================== CSS =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(45deg, #0c0d12, #10121c, #0e0f15, #14151d);
    background-size: 400% 400%;
    animation: gradientShift 30s ease infinite;
    color: #e7e7e7;
}

@keyframes gradientShift {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.card, .plot-card {
    background: rgba(30,30,38,0.55);
    backdrop-filter: blur(18px) saturate(180%);
    border-radius: 22px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 45px rgba(0,0,0,0.85);
    transition: all 0.35s ease;
}

.stButton button {
    border-radius: 18px;
    padding: 12px 22px;
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg,#2f323c,#8a8f9a,#2f323c);
    box-shadow: inset 0 1px 2px rgba(255,255,255,0.45), 0 10px 26px rgba(0,0,0,0.85);
    transition: all 0.35s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 32px rgba(0,200,255,0.8), 0 18px 40px rgba(0,0,0,0.95);
}

.plot-card {
    padding: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===================== Layout =====================
st.markdown("<div class='card'><h2>🎯 CAPTCHA + Refinement Live Preview</h2></div>", unsafe_allow_html=True)
control_col, live_col = st.columns([1,2])

# ===================== Controls =====================
with control_col:
    noise = st.slider("Noise", 0.0, 1.0, 0.25)
    distortion = st.slider("Distortion", 0.0, 1.0, 0.25)
    clutter = st.slider("Clutter", 0.0, 1.0, 0.25)
    target = st.selectbox("Target Difficulty", ["easy", "medium", "hard"])
    gen_btn = st.button("🎲 Generate CAPTCHA")
    auto_btn = st.button("🚀 Auto-Refine")

# ===================== Live Slots =====================
with live_col:
    img_slot = st.empty()
    plot1_slot, plot2_slot = st.columns(2)
    stats_slot = st.empty()

# ===================== Logic =====================
if gen_btn:
    img, text = generate_captcha(noise, distortion, clutter)
    pred, conf = predict(img)
    img_slot.image(img, use_column_width=True)
    stats_slot.markdown(f"**Text:** `{text}`  \n**Difficulty:** `{pred.upper()}`  \n**Confidence:** `{conf:.2f}`")

if auto_btn:
    confs = []
    for step in range(6):
        mat = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                img, _, _ = refine(target)
                _, c = predict(img)
                mat[i,j] = c

        confs.append(mat.mean())
        img_slot.image(img, use_column_width=True)

        # Convergence plot
        with plot1_slot:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(4,3))
            ax1.plot(confs, marker='o', color='#00ffff', linewidth=2)
            ax1.set_ylim(0,1)
            ax1.set_title("Convergence", color='white')
            st.pyplot(fig1)
            plt.close(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        # Heatmap plot
        with plot2_slot:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            ax2.set_title("Confidence Heatmap", color='white')
            st.pyplot(fig2)
            plt.close(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        time.sleep(0.5)

    st.success("Target difficulty stabilized ✔")
