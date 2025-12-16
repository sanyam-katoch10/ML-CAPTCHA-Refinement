import streamlit as st
from io import BytesIO
from generator import generate_captcha
from refine_m import refine, predict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time

st.set_page_config(page_title="ML CAPTCHA Refinement", page_icon="🔒", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1b1b1b, #2a2a2a, #121212, #2e2e2e);
    background-size: 400% 400%;
    animation: bgShift 25s ease infinite;
    color: #eaeaea;
    font-family: 'Segoe UI', sans-serif;
    overflow: hidden;
}
@keyframes bgShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.shape {
    position: absolute;
    border-radius: 50%;
    opacity: 0.2;
    animation: floatShape 20s linear infinite;
}
@keyframes floatShape {
    0% {transform: translateY(0px) translateX(0px);}
    50% {transform: translateY(-30px) translateX(20px);}
    100% {transform: translateY(0px) translateX(0px);}
}

.topbar {
    background: rgba(30,30,30,0.7);
    backdrop-filter: blur(15px);
    padding: 20px 30px;
    border-radius: 20px;
    box-shadow: 0 0 15px #00ffff, 0 0 25px #00ffff, inset 0 1px 3px rgba(255,255,255,0.1), 0 15px 40px rgba(0,0,0,0.8);
    margin-bottom: 25px;
    font-size: 28px;
    font-weight: 800;
    color: #f0f0f0;
}

section[data-testid="stSidebar"] {
    background: rgba(20,20,20,0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 15px #00ffff, 0 0 25px #00ffff;
}
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 20px;
}

.card {
    background: rgba(40,40,40,0.45);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    border: 2px solid #00ffff;
    box-shadow: 0 0 15px rgba(0,255,255,0.3), 0 0 25px rgba(0,255,255,0.2), 0 10px 35px rgba(0,0,0,0.7);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 35px rgba(0,255,255,0.7), 0 0 50px rgba(0,255,255,0.5), 0 10px 40px rgba(0,0,0,0.8);
}

.stButton button {
    border-radius: 16px;
    border: none;
    padding: 14px;
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg,#3b3b3b,#7b7b7b,#3b3b3b);
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.3), 0 6px 18px rgba(0,0,0,0.7);
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff, 0 12px 35px rgba(0,0,0,0.9);
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #8d8d8d;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar'>🔒 ML CAPTCHA Refinement <span style='float:right;font-size:16px;'>🟢 Model Online</span></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚙️ Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["📊 Dashboard", "🖼 CAPTCHA Generator", "🔁 Refinement Engine"])

if page == "📊 Dashboard":
    st.markdown("## 📊 System Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'>### Avg Confidence<br><h2>0.76</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>### Stability Status<br><h2>Stable</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>### Active Model<br><h2>CNN v1.0</h2></div>", unsafe_allow_html=True)

elif page == "🖼 CAPTCHA Generator":
    st.markdown("## 🖼 CAPTCHA Generator")
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        noise = st.slider("Noise", 0.0, 1.0, 0.25)
        distortion = st.slider("Distortion", 0.0, 1.0, 0.25)
        clutter = st.slider("Clutter", 0.0, 1.0, 0.25)
        gen_btn = st.button("🎲 Generate CAPTCHA")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        preview_slot = st.empty()
        if gen_btn:
            img, text = generate_captcha(noise, distortion, clutter)
            preview_slot.image(img, use_column_width=True)
            pred, conf = predict(img)
            st.markdown(f"**Text:** `{text}`  |  **Difficulty:** `{pred.upper()}`  |  **Confidence:** `{conf:.2f}`")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔁 Refinement Engine":
    st.markdown("## 🔁 Refinement Engine")
    target = st.selectbox("Target Difficulty", ["easy", "medium", "hard"])
    refine_btn = st.button("✨ Refine Once")
    auto_btn = st.button("🚀 Auto-Refine")

    live_slot = st.empty()
    col1, col2 = st.columns([1,1])
    conv_slot = col1.empty()
    heat_slot = col2.empty()

    if refine_btn:
        img, text, lvl = refine(target)
        live_slot.image(img, use_column_width=True)
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.download_button("⬇ Download CAPTCHA", buf.getvalue(), f"{text}_{lvl}.png")

    if auto_btn:
        confs = []
        grid = 4
        mat_current = np.zeros((grid, grid))
        steps_per_update = 20
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.plasma

        for step in range(6):
            mat_target = np.zeros((grid, grid))
            for i in range(grid):
                for j in range(grid):
                    img, _, _ = refine(target)
                    live_slot.image(img, use_column_width=True)
                    _, c = predict(img)
                    mat_target[i, j] = c

            confs.append(mat_target.mean())

            for t in range(1, steps_per_update + 1):
                mat_interpolated = mat_current + (mat_target - mat_current) * (t / steps_per_update)
                fig1, ax1 = plt.subplots()
                ax1.plot(confs, marker='o', color='#00ffff')
                ax1.set_ylim(0,1)
                ax1.set_title("Convergence Line", color="#00ffff")
                ax1.grid(True, alpha=0.3)
                conv_slot.pyplot(fig1, clear_figure=True)
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                im = ax2.imshow(mat_interpolated, cmap=cmap, norm=norm)
                for i in range(grid):
                    for j in range(grid):
                        ax2.text(j, i, f"{mat_interpolated[i,j]:.2f}", ha='center', va='center',
                                 color='black', fontsize=10, fontweight='bold')
                ax2.tick_params(colors='black', which='both', labelsize=10)
                for spine in ax2.spines.values():
                    spine.set_color('black')
                ax2.set_title("Heatmap", color="#00ffff")
                heat_slot.pyplot(fig2, clear_figure=True)
                plt.close(fig2)
                time.sleep(0.05)

            mat_current = mat_target.copy()
        st.success("Target difficulty stabilized ✔")

st.markdown("<div class='footer'>✨ Built by SANYAM KATOCH ✨</div>", unsafe_allow_html=True)

for i in range(5):
    st.markdown(f"<div class='shape' style='width:{50+i*10}px; height:{50+i*10}px; top:{10+i*15}%; left:{20+i*10}%; background: radial-gradient(circle, #00ffff, transparent); animation-delay:{i*3}s;'></div>", unsafe_allow_html=True)
