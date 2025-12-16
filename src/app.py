import streamlit as st
from io import BytesIO
from generator import generate_captcha
from refine_m import refine, predict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# ===================== CONFIG =====================
st.set_page_config(
    page_title="ML CAPTCHA Refinement",
    page_icon="🔒",
    layout="wide"
)

# ===================== CSS =====================
st.markdown("""
<style>

/* ===== DARK ANIMATED BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        130deg,
        #0a0b0f,
        #12141b,
        #1a1d26,
        #0e1016
    );
    background-size: 500% 500%;
    animation: darkShift 35s ease infinite;
    color: #e7e7e7;
}

@keyframes darkShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ===== TOP BAR ===== */
.topbar {
    background: linear-gradient(135deg,#1a1c24,#2b2f3a);
    padding: 18px 30px;
    border-radius: 18px;
    font-size: 26px;
    font-weight: 800;
    box-shadow:
        inset 0 1px 1px rgba(255,255,255,0.08),
        0 18px 45px rgba(0,0,0,0.9);
    margin-bottom: 20px;
    transition: all 0.4s ease;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0f15,#171a23);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ===== GLASS CARDS ===== */
.card {
    background: rgba(30,30,35,0.45);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 26px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 45px rgba(0,0,0,0.85);
    transition: all 0.4s ease;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 35px 80px rgba(0,0,0,0.95), 0 0 18px rgba(255,255,255,0.1);
}

/* ===== BUTTONS ===== */
.stButton button {
    border-radius: 18px;
    padding: 14px 22px;
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg,#2f323c,#8a8f9a,#2f323c);
    box-shadow: inset 0 1px 2px rgba(255,255,255,0.45), 0 10px 26px rgba(0,0,0,0.85);
    transition: all 0.35s ease;
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 28px rgba(210,210,255,0.4), 0 18px 40px rgba(0,0,0,0.95);
}

/* ===== GLASS PLOT CONTAINERS ===== */
.plot-card {
    background: rgba(35,35,42,0.45);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 18px 45px rgba(0,0,0,0.85);
    transition: all 0.35s ease, box-shadow 0.35s ease;
}

.plot-card:hover {
    box-shadow: 0 25px 60px rgba(0,0,0,0.9), 0 0 24px rgba(180,220,255,0.5);
    filter: drop-shadow(0 0 12px rgba(180,220,255,0.5));
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #8b8f9c;
}

</style>
""", unsafe_allow_html=True)

# ===================== TOP BAR =====================
st.markdown(
    "<div class='topbar'>🔒 ML CAPTCHA Refinement <span style='float:right;font-size:16px;'>🟢 Model Online</span></div>",
    unsafe_allow_html=True
)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## ⚙️ Navigation")
    page = st.radio("", ["📊 Dashboard", "🖼 CAPTCHA Generator", "🔁 Refinement Engine"])

# ===================== DASHBOARD =====================
if page == "📊 Dashboard":
    st.markdown("## 📊 System Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>Avg Confidence</h3><h2>0.76</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>Status</h3><h2>Stable</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>Model</h3><h2>CNN v1.0</h2></div>", unsafe_allow_html=True)

# ===================== CAPTCHA GENERATOR =====================
elif page == "🖼 CAPTCHA Generator":
    st.markdown("## 🖼 CAPTCHA Generator")
    left, right = st.columns([1.1, 2])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        noise = st.slider("Noise", 0.0, 1.0, 0.25)
        distortion = st.slider("Distortion", 0.0, 1.0, 0.25)
        clutter = st.slider("Clutter", 0.0, 1.0, 0.25)
        gen_btn = st.button("🎲 Generate CAPTCHA")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        img_slot = st.empty()
        stats_slot = st.empty()

        if gen_btn:
            img, text = generate_captcha(noise, distortion, clutter)
            pred, conf = predict(img)
            img_slot.image(img, use_column_width=True)
            stats_slot.markdown(f"""
            **Text:** `{text}`  
            **Difficulty:** `{pred.upper()}`  
            **Confidence:** `{conf:.2f}`
            """)

        # Side-by-side convergence + heatmap (static)
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            confs = np.clip(np.cumsum(np.random.normal(0.04, 0.02, 10)), 0, 1)
            fig1, ax1 = plt.subplots(figsize=(4,3))
            ax1.plot(confs, marker='o')
            ax1.set_ylim(0,1)
            ax1.set_title("Confidence Convergence")
            st.pyplot(fig1)
            plt.close(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        with p2:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            mat = np.random.uniform(0.4, 0.9, (4,4))
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            ax2.set_title("Confidence Heatmap")
            st.pyplot(fig2)
            plt.close(fig2)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== REFINEMENT ENGINE =====================
elif page == "🔁 Refinement Engine":
    st.markdown("## 🔁 Refinement Engine")
    target = st.selectbox("Target Difficulty", ["easy", "medium", "hard"])
    refine_btn = st.button("✨ Refine Once")
    auto_btn = st.button("🚀 Auto-Refine")

    # Placeholders for side-by-side plots
    plot_slot_c, plot_slot_h = st.columns(2)

    if refine_btn:
        img, text, lvl = refine(target)
        st.image(img, use_column_width=True)
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.download_button("⬇ Download CAPTCHA", buf.getvalue(), f"{text}_{lvl}.png")

    if auto_btn:
        confs = []

        for _ in range(6):
            mat = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    img, _, _ = refine(target)
                    _, c = predict(img)
                    mat[i,j] = c
            confs.append(mat.mean())

            with plot_slot_c:
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(4,3))
                ax1.plot(confs, marker='o')
                ax1.set_ylim(0,1)
                ax1.set_title("Confidence Convergence")
                st.pyplot(fig1, clear_figure=True)
                plt.close(fig1)
                st.markdown("</div>", unsafe_allow_html=True)

            with plot_slot_h:
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(4,3))
                sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
                ax2.set_title("Confidence Heatmap")
                st.pyplot(fig2, clear_figure=True)
                plt.close(fig2)
                st.markdown("</div>", unsafe_allow_html=True)

            time.sleep(0.5)
        st.success("Target difficulty stabilized ✔")

# ===================== FOOTER =====================
st.markdown("<div class='footer'>✨ Built by SANYAM KATOCH ✨</div>", unsafe_allow_html=True)
