# app.py — GrapeAI Intelligence System
# Unified Disease Decision Engine edition

import re
import streamlit as st
import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from disease_model      import load_disease_model, predict_disease
from canopy_analysis    import calculate_canopy_coverage, interpret_canopy
from suggestions        import get_suggestions, get_urgency_color
from severity_estimator import estimate_severity
from leaf_features      import extract_leaf_features
from model_info         import print_model_info, MODEL_REGISTRY, DATASET_STATS
from grape_ai_engine    import analyze_grape_image
from disease_engine     import (
    detect_image_type,
    parse_gemini_output,
    resolve_diagnosis,
    get_source_badge,
    LEAF_DISEASE_NAMES,
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.set_page_config(
    page_title="GrapeAI — Crop Intelligence System",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

print_model_info()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1a0f 0%, #1a1a2e 50%, #0f1a0f 100%); color: #e8f5e8; }
.login-wrap { max-width:420px; margin:6rem auto; padding:2.5rem 2rem; background:rgba(255,255,255,.04); border:1px solid rgba(107,142,35,.3); border-radius:20px; box-shadow:0 24px 64px rgba(0,0,0,.4); }
.login-logo  { text-align:center; font-size:3.5rem; margin-bottom:.5rem; }
.login-title { font-family:'DM Serif Display',serif; font-size:2rem; color:#a8d5a2; text-align:center; margin:0 0 .3rem; }
.login-sub   { color:#6b8e6b; text-align:center; font-size:.85rem; margin-bottom:2rem; }
.main-header { text-align:center; padding:1.6rem 0 1rem; border-bottom:1px solid rgba(107,142,35,.3); margin-bottom:2rem; }
.main-header h1 { font-family:'DM Serif Display',serif; font-size:2.8rem; color:#a8d5a2; letter-spacing:-1px; margin:0; }
.main-header p  { color:#6b8e6b; font-size:.95rem; font-weight:300; margin-top:.4rem; }
.metric-card { background:rgba(255,255,255,.04); border:1px solid rgba(107,142,35,.25); border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.metric-label { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; color:#6b8e6b; margin-bottom:.3rem; }
.metric-value { font-family:'DM Serif Display',serif; font-size:2.1rem; color:#a8d5a2; line-height:1; }
.metric-sub   { font-size:.78rem; color:#6b8e6b; margin-top:.3rem; }
.engine-banner { border-radius:10px; padding:.9rem 1.2rem; margin-bottom:1rem; border-left:4px solid; }
.engine-cnn        { background:rgba(27,94,32,.12);   border-color:#2e7d32; }
.engine-gemini     { background:rgba(21,101,192,.12);  border-color:#1565c0; }
.engine-fusion     { background:rgba(106,27,154,.12);  border-color:#6a1b9a; }
.engine-uncertain  { background:rgba(230,81,0,.12);    border-color:#e65100; }
.engine-none       { background:rgba(69,90,100,.12);   border-color:#455a64; }
.image-type-pill { display:inline-block; border-radius:50px; padding:.25rem .85rem; font-size:.75rem; font-weight:600; letter-spacing:.5px; margin-bottom:.8rem; }
.pill-leaf  { background:rgba(33,197,93,.15);  color:#21c55d; border:1px solid rgba(33,197,93,.4); }
.pill-bunch { background:rgba(124,77,255,.15); color:#b39ddb; border:1px solid rgba(124,77,255,.4); }
.pill-ambig { background:rgba(255,160,0,.15);  color:#ffcc02; border:1px solid rgba(255,160,0,.4); }
.suggestion-box   { background:rgba(255,255,255,.03); border-left:3px solid #6b8e6b; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin-bottom:.8rem; }
.suggestion-title { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:#6b8e6b; margin-bottom:.6rem; }
.suggestion-item  { font-size:.87rem; color:#c8dcc8; padding:.25rem 0 .25rem 1rem; border-left:2px solid rgba(107,142,35,.3); margin-bottom:.4rem; }
.feat-row { display:flex; justify-content:space-between; border-bottom:1px solid rgba(107,142,35,.1); padding:.4rem 0; font-size:.82rem; }
.feat-key { color:#6b8e6b; }
.feat-val { color:#a8d5a2; font-weight:500; }
.urgency-HIGH   { background:rgba(255,75,75,.1);   border:1px solid rgba(255,75,75,.4);   border-radius:8px; padding:.8rem 1rem; color:#ff8888; }
.urgency-MEDIUM { background:rgba(255,165,0,.1);   border:1px solid rgba(255,165,0,.4);   border-radius:8px; padding:.8rem 1rem; color:#ffcc66; }
.urgency-LOW    { background:rgba(33,197,93,.1);   border:1px solid rgba(33,197,93,.4);   border-radius:8px; padding:.8rem 1rem; color:#88ff99; }
.arch-table { width:100%; border-collapse:collapse; font-size:.82rem; }
.arch-table th { color:#6b8e6b; font-weight:600; text-align:left; padding:.4rem .6rem; border-bottom:1px solid rgba(107,142,35,.3); }
.arch-table td { color:#c8dcc8; padding:.35rem .6rem; border-bottom:1px solid rgba(107,142,35,.1); }
.arch-table tr:last-child td { border-bottom:none; }
.stat-card { background:rgba(255,255,255,.04); border:1px solid rgba(107,142,35,.2); border-radius:14px; padding:1.4rem 1.2rem; text-align:center; }
.stat-icon  { font-size:2rem; margin-bottom:.5rem; }
.stat-val   { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#a8d5a2; }
.stat-label { font-size:.75rem; color:#6b8e6b; font-weight:500; text-transform:uppercase; letter-spacing:1px; margin-top:.3rem; }
.section-divider { border:none; border-top:1px solid rgba(107,142,35,.15); margin:1.5rem 0; }
.footer { text-align:center; padding:2rem 0 1rem; color:#3d5c3d; font-size:.76rem; border-top:1px solid rgba(107,142,35,.1); margin-top:3rem; }
.gemini-box { background:rgba(21,101,192,.07); border:1px solid rgba(21,101,192,.25); border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.gemini-title { font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:#90caf9; margin-bottom:.8rem; }
.gemini-body  { font-size:.86rem; color:#bbdefb; line-height:1.7; white-space:pre-wrap; }
</style>
""", unsafe_allow_html=True)

UNET_TABLE = """
<table class="arch-table">
<tr><th>Layer block</th><th>Details</th></tr>
<tr><td>Input</td><td>572 x 572 x 3 RGB</td></tr>
<tr><td>Encoder x4</td><td>Conv → BN → ReLU, MaxPool</td></tr>
<tr><td>Bottleneck</td><td>1024-channel feature map</td></tr>
<tr><td>Decoder x4</td><td>TranspConv + skip concat</td></tr>
<tr><td>Output</td><td>1-ch sigmoid mask</td></tr>
<tr><td>Loss</td><td>BCE + Dice</td></tr>
<tr><td>Parameters</td><td>~31 M</td></tr>
</table>"""

CNN_TABLE = """
<table class="arch-table">
<tr><th>Layer</th><th>Output shape</th></tr>
<tr><td>EfficientNet-B4 backbone</td><td>1792-dim features</td></tr>
<tr><td>Dropout(0.3)</td><td>—</td></tr>
<tr><td>Linear(1792 → 512)</td><td>512-dim</td></tr>
<tr><td>BatchNorm + ReLU</td><td>512-dim</td></tr>
<tr><td>Dropout(0.2)</td><td>—</td></tr>
<tr><td>Linear(512 → 128)</td><td>128-dim</td></tr>
<tr><td>Linear(128 → 4)</td><td>4 disease classes</td></tr>
</table>"""

@st.cache_resource(show_spinner=False)
def get_disease_model():
    custom = Path("checkpoints/grape_disease.pth")
    return load_disease_model(str(custom) if custom.exists() else None)

def run_unet(input_path, output_path):
    ckpt = "checkpoints/checkpoint_epoch30.pth"
    if not Path(ckpt).exists():
        return False
    try:
        r = subprocess.run(
            [sys.executable, "predict.py", "-m", ckpt, "-i", input_path, "-o", output_path],
            capture_output=True, text=True, timeout=60
        )
        return r.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False

def fallback_mask(image, output_path):
    img = np.array(image.convert("RGB").resize((256, 256))).astype(float)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (g > r * 1.1) & (g > b * 1.1) & (g > 50)
    Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)

def fig_overlay(original, mask):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0f1a0f")
    for ax in axes:
        ax.set_facecolor("#0f1a0f"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#2d4a2d")
    orig     = original.convert("RGB").resize((256, 256))
    msk      = np.array(mask.convert("L").resize((256, 256)))
    orig_arr = np.array(orig)
    overlay  = orig_arr.copy()
    tint     = np.zeros_like(orig_arr); tint[:, :, 1] = 120
    canopy   = msk > 127
    overlay[canopy] = (overlay[canopy] * .5 + tint[canopy] * .5).astype(np.uint8)
    axes[0].imshow(orig);               axes[0].set_title("Input image",  color="#a8d5a2", fontsize=11, pad=10)
    axes[1].imshow(msk, cmap="Greens"); axes[1].set_title("Canopy mask",  color="#a8d5a2", fontsize=11, pad=10)
    axes[2].imshow(overlay)
    axes[2].legend(handles=[mpatches.Patch(color="#50c878", label="Detected canopy")],
                   loc="lower right", facecolor="#1a2e1a", edgecolor="#2d4a2d", labelcolor="#a8d5a2", fontsize=8)
    axes[2].set_title("Overlay", color="#a8d5a2", fontsize=11, pad=10)
    plt.tight_layout(pad=1.5); return fig

def fig_disease_bars(probs):
    if not probs: return None
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(probs) * 0.7)))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    labels  = list(probs.keys())
    values  = [v * 100 for v in probs.values()]
    palette = ["#FF6B6B","#FF9F43","#FFC947","#21C55D","#64B5F6","#BA68C8","#4DB6AC","#FFD54F"]
    colors  = palette[:len(labels)]
    bars    = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left", color="#a8d5a2", fontsize=9)
    ax.set_xlim(0, 115); ax.tick_params(colors="#a8d5a2", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#2d4a2d")
    plt.tight_layout(); return fig

def fig_training_curves():
    try:
        with open("training_history.json","r",encoding="utf-8") as f: hist = json.load(f)
    except FileNotFoundError: return None
    fig = plt.figure(figsize=(14,5)); fig.patch.set_facecolor("#0f1a0f")
    gs  = GridSpec(1,2,figure=fig,wspace=0.35)
    def sax(ax,title):
        ax.set_facecolor("#0f1a0f"); ax.set_title(title,color="#a8d5a2",fontsize=11,pad=10)
        ax.tick_params(colors="#6b8e6b",labelsize=8)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color("#2d4a2d")
        ax.xaxis.label.set_color("#6b8e6b"); ax.yaxis.label.set_color("#6b8e6b")
    unet=hist["canopy_unet"]; ax1=fig.add_subplot(gs[0])
    ax1.plot(unet["epochs"],unet["train_loss"],color="#5DCAA5",lw=1.8,label="Train loss")
    ax1.plot(unet["epochs"],unet["val_loss"],color="#FF9F43",lw=1.8,linestyle="--",label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(facecolor="#1a2e1a",edgecolor="#2d4a2d",labelcolor="#a8d5a2",fontsize=8)
    sax(ax1,"GrapeCanopyNet — training loss (30 epochs)")
    cnn=hist["disease_cnn"]; ax2=fig.add_subplot(gs[1])
    ax2.plot(cnn["epochs"],[v*100 for v in cnn["train_acc"]],color="#5DCAA5",lw=1.8,label="Train acc")
    ax2.plot(cnn["epochs"],[v*100 for v in cnn["val_acc"]],color="#FF9F43",lw=1.8,linestyle="--",label="Val acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(facecolor="#1a2e1a",edgecolor="#2d4a2d",labelcolor="#a8d5a2",fontsize=8)
    sax(ax2,"GrapeLeafCNN — validation accuracy (20 epochs)")
    plt.tight_layout(); return fig

USERS = {"admin": "admin123", "revan": "grape2025", "guest": "guest123"}

def login_page():
    st.markdown("""
<div class="login-wrap">
    <div class="login-logo">🍇</div>
    <div class="login-title">GrapeAI</div>
    <div class="login-sub">Crop Intelligence System &middot; B.Tech Final Year Project</div>
</div>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("👤  Username", placeholder="Enter username")
        password = st.text_input("🔒  Password", type="password", placeholder="Enter password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True, type="primary"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.rerun()
            elif not username or not password:
                st.warning("Please enter both username and password.")
            else:
                st.error("❌  Invalid credentials.")
        st.markdown("""
<div style="text-align:center;margin-top:1.2rem;font-size:.75rem;color:#3d5c3d">
Demo: admin / admin123
</div>""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
<div style="background:rgba(107,142,35,.12);border:1px solid rgba(107,142,35,.3);
            border-radius:10px;padding:.8rem 1rem;margin-bottom:1rem">
    <div style="font-size:.72rem;color:#6b8e6b;text-transform:uppercase;letter-spacing:1px;margin-bottom:.2rem">Logged in as</div>
    <div style="font-size:1rem;font-weight:600;color:#a8d5a2">👤 {st.session_state.username.capitalize()}</div>
</div>""", unsafe_allow_html=True)
        if st.button("⏻  Logout", use_container_width=True):
            st.session_state.logged_in = False; st.session_state.username = ""; st.rerun()
        st.markdown("---")
        page = st.radio("Navigation",
                        ["🏠 Dashboard","🔬 Analysis","📊 Model Registry","📚 Dataset Info","ℹ️ About"],
                        label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### ⚙️  Settings")
        use_unet      = st.checkbox("Use U-Net (canopy)",         value=True)
        show_overlay  = st.checkbox("Show segmentation overlay",   value=True)
        show_probs    = st.checkbox("Show disease probabilities",   value=True)
        show_features = st.checkbox("Show leaf feature analysis",   value=True)
        show_severity = st.checkbox("Show severity estimation",     value=True)
        show_training = st.checkbox("Show training curves",         value=True)
        show_gemini   = st.checkbox("Show Canopy AI raw output",       value=False)
    return page, dict(use_unet=use_unet, show_overlay=show_overlay, show_probs=show_probs,
                      show_features=show_features, show_severity=show_severity,
                      show_training=show_training, show_gemini=show_gemini)

def page_dashboard():
    st.markdown("""
<div class="main-header">
    <h1>🍇 GrapeAI Intelligence System</h1>
    <p>AI-Based Crop Canopy Analysis &amp; Grape Disease Detection &middot; B.Tech Final Year Project</p>
</div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(icon,val,label) in zip([c1,c2,c3,c4],[
        ("🌿","U-Net","Canopy Model"),("🧬","EfficientNet-B4","Disease Model"),
        ("🍇","4 Classes","Leaf Diseases"),("🤖","Canopy AI","Expert Analysis")]):
        col.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
                     f'<div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Unified Disease Decision Engine")
    st.markdown("""
<div style="background:rgba(255,255,255,.03);border:1px solid rgba(107,142,35,.2);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1.5rem">
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem">
<div><div style="font-size:.72rem;color:#6b8e6b;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem">Step 1 — Image Routing</div>
<div style="font-size:.84rem;color:#c8dcc8;line-height:1.7">Detects whether image is a <b style="color:#a8d5a2">leaf</b>, <b style="color:#b39ddb">bunch</b>, or ambiguous. Routing decides which model is trusted.</div></div>
<div><div style="font-size:.72rem;color:#6b8e6b;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem">Step 2 — Parallel Inference</div>
<div style="font-size:.84rem;color:#c8dcc8;line-height:1.7"><b style="color:#a8d5a2">CNN</b> runs on all images. <b style="color:#90caf9">Canopy AI</b> runs in parallel. For bunch images CNN is marked not applicable.</div></div>
<div><div style="font-size:.72rem;color:#6b8e6b;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem">Step 3 — Conflict Resolution</div>
<div style="font-size:.84rem;color:#c8dcc8;line-height:1.7">Leaf → CNN preferred unless Canopy AI ≥80% confidence. Bunch → Gemini only. Severity blended from Canopy AI label + pixel analysis.</div></div>
</div></div>""", unsafe_allow_html=True)
    st.markdown("### 🔄 Pipeline Flow")
    steps=[("📤","Upload","JPG/PNG image"),("🔍","Image Type","Leaf/Bunch detector"),
           ("🔲","Canopy","U-Net segmentation"),("🧬","CNN","4-class classification"),
           ("🤖","Gemini","Vision AI analysis"),("⚖️","Resolve","Decision engine"),("📊","Output","Dashboard + report")]
    cols=st.columns(len(steps))
    for col,(icon,title,desc) in zip(cols,steps):
        col.markdown(f'<div style="text-align:center;padding:.8rem .4rem;background:rgba(255,255,255,.03);border-radius:10px;border:1px solid rgba(107,142,35,.18)">'
                     f'<div style="font-size:1.6rem">{icon}</div>'
                     f'<div style="font-size:.78rem;font-weight:600;color:#a8d5a2;margin:.3rem 0 .2rem">{title}</div>'
                     f'<div style="font-size:.65rem;color:#4a6b4a;line-height:1.4">{desc}</div></div>', unsafe_allow_html=True)

def page_model_registry():
    st.markdown("## 🔬 Model Registry")
    for model_name,info in MODEL_REGISTRY.items():
        with st.expander(f"📦 {model_name}", expanded=True):
            c1,c2=st.columns(2); items=list(info.items()); half=len(items)//2
            with c1:
                for k,v in items[:half]: st.markdown(f"**{k}:** {v}")
            with c2:
                for k,v in items[half:]: st.markdown(f"**{k}:** {v}")
    st.markdown("---"); st.markdown("#### 📐 Architecture Details")
    col1,col2=st.columns(2)
    with col1: st.markdown("**GrapeCanopyNet (U-Net)**"); st.markdown(UNET_TABLE, unsafe_allow_html=True)
    with col2: st.markdown("**GrapeLeafCNN**"); st.markdown(CNN_TABLE, unsafe_allow_html=True)

def page_dataset():
    st.markdown("## 📊 Dataset Statistics")
    for ds_name,stats in DATASET_STATS.items():
        with st.expander(f"📁 {ds_name}", expanded=True):
            for k,v in stats.items(): st.markdown(f"**{k}:** {v}")

def page_about():
    st.markdown("## ℹ️ About GrapeAI")
    st.markdown("""
<div style="background:rgba(255,255,255,.03);border:1px solid rgba(107,142,35,.2);border-radius:12px;padding:1.6rem 2rem">
<p style="color:#c8dcc8;line-height:1.8;font-size:.9rem"><b style="color:#a8d5a2">GrapeAI</b> is a final year B.Tech project integrating two deep learning models and Canopy AI into a unified grape disease diagnosis pipeline.</p>
<br>
<table style="width:100%;border-collapse:collapse;font-size:.86rem">
<tr><td style="color:#6b8e6b;padding:.4rem 0;width:200px"><b>Project Title</b></td><td style="color:#c8dcc8">AI-Based Crop Canopy and Grape Disease Analysis System</td></tr>
<tr><td style="color:#6b8e6b;padding:.4rem 0"><b>Department</b></td><td style="color:#c8dcc8">Electronics &amp; Telecommunication Engineering</td></tr>
<tr><td style="color:#6b8e6b;padding:.4rem 0"><b>Framework</b></td><td style="color:#c8dcc8">PyTorch 2.x · Streamlit · Google Canopy AI</td></tr>
<tr><td style="color:#6b8e6b;padding:.4rem 0"><b>Canopy Model</b></td><td style="color:#c8dcc8">GrapeCanopyNet — U-Net, 587 images, 30 epochs</td></tr>
<tr><td style="color:#6b8e6b;padding:.4rem 0"><b>Disease Model</b></td><td style="color:#c8dcc8">GrapeLeafCNN — EfficientNet-B4, PlantVillage</td></tr>
<tr><td style="color:#6b8e6b;padding:.4rem 0"><b>Decision Engine</b></td><td style="color:#c8dcc8">Image-type routing + conflict resolver + severity blender</td></tr>
</table></div>""", unsafe_allow_html=True)

def page_analysis(settings):
    st.markdown("""
<div class="main-header">
    <h1>🔬 Grape Analysis</h1>
    <p>Upload any grape image — leaf or bunch — the engine routes automatically</p>
</div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a grape image (leaf or bunch)",
                                     type=["jpg","jpeg","png"])
    if not uploaded_file:
        st.markdown("""
<div style="text-align:center;padding:3rem;border:2px dashed rgba(107,142,35,.3);border-radius:16px;margin:2rem 0;background:rgba(107,142,35,.03)">
    <div style="font-size:3rem;margin-bottom:1rem">🍃</div>
    <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#6b8e6b;margin-bottom:.5rem">Upload a grape leaf or bunch image to begin</div>
    <div style="font-size:.82rem;color:#3d5c3d">The decision engine automatically detects image type and selects the correct AI source</div>
</div>""", unsafe_allow_html=True)
        return

    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.TemporaryDirectory() as tmp:
        inp       = os.path.join(tmp, "input.jpg")
        mask_path = os.path.join(tmp, "mask.png")
        image.save(inp, "JPEG")

        prog = st.progress(0, "🔄  Initialising pipeline…")
        prog.progress(8,  "🔍  Detecting image type…");      img_type = detect_image_type(image)
        prog.progress(15, "🧬  Loading GrapeLeafCNN…");      cnn_model = get_disease_model()
        prog.progress(28, "🔲  Running canopy segmentation…")
        unet_ok = run_unet(inp, mask_path) if settings["use_unet"] else False
        if not unet_ok:
            prog.progress(35, "🌿  Vegetation index fallback…"); fallback_mask(image, mask_path)
        prog.progress(44, "📊  Calculating canopy coverage…")
        canopy = calculate_canopy_coverage(mask_path); c_info = interpret_canopy(canopy["coverage_percent"])
        prog.progress(55, "🍇  Running CNN classification…")
        cnn_class, cnn_display, cnn_conf, cnn_probs = predict_disease(cnn_model, image)
        prog.progress(65, "📈  Pixel severity analysis…");    pixel_severity = estimate_severity(image, cnn_class)
        prog.progress(72, "🔬  Extracting leaf features…");   features = extract_leaf_features(image)
        prog.progress(82, "🤖  Canopy AI analysis…")
        try:    gemini_raw = analyze_grape_image(inp)
        except Exception as e: gemini_raw = f"Canopy AI Error: {e}"
        prog.progress(90, "⚖️  Unifying diagnosis…")
        gemini_parsed = parse_gemini_output(gemini_raw)
        diagnosis = resolve_diagnosis(img_type, cnn_class, cnn_conf, cnn_probs, gemini_parsed, pixel_severity)
        prog.progress(100, "✅  Analysis complete!"); prog.empty()

        final_disease  = diagnosis["disease_name"]
        final_conf     = diagnosis["confidence"]
        final_probs    = diagnosis["probs"]
        final_severity = diagnosis["severity"]
        final_urgency  = diagnosis["urgency"]
        source         = diagnosis["source"]
        reason         = diagnosis["reason"]
        badge_html, badge_label = get_source_badge(source)

        try:
            suggestions = get_suggestions(cnn_class if diagnosis["is_leaf_disease"] else "Grape___healthy")
        except Exception:
            suggestions = get_suggestions("Grape___healthy")

        if not diagnosis["is_leaf_disease"] and gemini_parsed.get("disease_name"):
            ai_note = gemini_parsed.get("symptoms") or f"{gemini_parsed.get('disease_name')} detected on grape cluster."
            if len(ai_note) > 150: 
                ai_note = ai_note[:147] + "..."
            suggestions["severity_note"] = f"Canopy AI Expert Note: {ai_note}"

        # OVERRIDE: If Gemini took control, use its dynamic text instead of the rigid suggestions file
        if not diagnosis["is_leaf_disease"] and gemini_parsed.get("disease_name"):
             ai_note = gemini_parsed.get("symptoms") or f"{gemini_parsed.get('disease_name')} detected on grape cluster."
             
             # Safely truncate to prevent UI overflow
             if len(ai_note) > 150: ai_note = ai_note[:147] + "..."
             
             suggestions["severity_note"] = f"AI Expert Note: {ai_note}"

        pill_class = {"leaf":"pill-leaf","bunch":"pill-bunch"}.get(img_type["type"],"pill-ambig")
        pill_label = {"leaf": f"🍃 Leaf Image ({img_type['confidence']:.0%})",
                      "bunch":f"🍇 Bunch Image ({img_type['confidence']:.0%})",
                      "ambiguous":f"⚠️ Ambiguous ({img_type['confidence']:.0%})"}[img_type["type"]]
        engine_cls = {"cnn":"engine-cnn","gemini":"engine-gemini","cnn+gemini":"engine-fusion",
                      "uncertain":"engine-uncertain","none":"engine-none"}.get(source,"engine-none")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## 📋 Analysis Results")
        st.markdown(f'<div class="image-type-pill {pill_class}">{pill_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="engine-banner {engine_cls}"><span style="font-size:.72rem;color:#aaa;text-transform:uppercase;letter-spacing:1px">Decision Engine</span> &nbsp; {badge_html} &nbsp; <span style="font-size:.84rem;color:#ddd">{reason}</span></div>', unsafe_allow_html=True)

        col_img, col_metrics = st.columns([1, 1])
        with col_img:
            st.markdown("**Input image**")
            st.image(image, use_container_width=True, caption=uploaded_file.name)
        with col_metrics:
            st.markdown(f'<div class="metric-card"><div class="metric-label">🌿 Canopy Coverage</div><div class="metric-value">{canopy["coverage_percent"]:.1f}%</div><div class="metric-sub">{c_info["icon"]} {c_info["label"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-label">🔬 Detected Condition &nbsp; {badge_html}</div><div style="font-family:\'DM Serif Display\',serif;font-size:1.35rem;color:#a8d5a2;line-height:1.2;margin:.3rem 0">{final_disease or "Unknown"}</div><div class="metric-sub">Confidence: {final_conf*100:.1f}%</div></div>', unsafe_allow_html=True)
            if settings["show_severity"]:
                sev=final_severity
                st.markdown(f'<div class="metric-card"><div class="metric-label">📈 Disease Severity <span style="font-size:.65rem;color:#888;margin-left:.5rem">src:{sev.get("source","—")}</span></div><div class="metric-value" style="color:{sev["severity_color"]}">{sev["severity_pct"]}%</div><div class="metric-sub">{sev["severity_label"]} &middot; Symptom area: {sev["symptom_area_pct"]}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="urgency-{final_urgency}"><b>Alert Level: {final_urgency}</b><br><span style="font-size:.82rem">{suggestions["severity_note"]}</span></div>', unsafe_allow_html=True)

        if img_type["type"] == "bunch":
            st.info("ℹ️  Bunch image detected. CNN is disabled (trained on leaves only). Diagnosis powered by Canopy AI.")

        with st.expander("🔍 CNN vs Canopy AI — Raw Comparison"):
            cc1,cc2=st.columns(2)
            with cc1:
                st.markdown("**GrapeLeafCNN (local)**")
                st.markdown(f"- Prediction: `{cnn_display}`\n- Confidence: `{cnn_conf*100:.1f}%`\n- Scope: leaf images only")
                if img_type["type"]=="bunch": st.warning("Not used — bunch image.")
            with cc2:
                st.markdown("**Canopy AI**")
                st.markdown(f"- Prediction: `{gemini_parsed.get('disease_name') or 'N/A'}`\n- Confidence: `{gemini_parsed.get('confidence_label') or 'N/A'}`\n- Severity: `{gemini_parsed.get('severity_label') or 'N/A'}`")
            st.markdown(f"**Engine decision:** {reason}")

        if settings["show_gemini"] and gemini_raw:
            st.markdown(f'<div class="gemini-box"><div class="gemini-title">🤖 Canopy AI — Full Expert Report</div><div class="gemini-body">{gemini_raw}</div></div>', unsafe_allow_html=True)

        if settings["show_overlay"]:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### 🔲 Canopy Segmentation — GrapeCanopyNet")
            try:
                mask_img=Image.open(mask_path); fig=fig_overlay(image, mask_img)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                method="GrapeCanopyNet (U-Net)" if unet_ok else "Vegetation index (fallback)"
                st.markdown(f'<div style="text-align:center;font-size:.77rem;color:#4a6b4a;margin-top:.5rem">Model: {method} &middot; Canopy pixels: {canopy["white_pixels"]:,} / {canopy["total_pixels"]:,}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="suggestion-box" style="margin-top:.8rem"><div class="suggestion-title">Canopy interpretation</div><div class="suggestion-item">{c_info["icon"]} {c_info["advice"]}</div></div>', unsafe_allow_html=True)
            except Exception as e: st.warning(f"Overlay error: {e}")

        if settings["show_probs"] and final_probs:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            src_label = "GrapeLeafCNN" if source in ("cnn","cnn+gemini") else "Canopy AI"
            st.markdown(f"#### 🧬 {src_label} — Classification Probabilities")
            col_p,col_c=st.columns([1,1])
            with col_p:
                for name,prob in final_probs.items():
                    ca,cb=st.columns([3,1])
                    ca.markdown(f"<span style='font-size:.82rem;color:#a8d5a2'>{name}</span>", unsafe_allow_html=True)
                    cb.markdown(f"<span style='font-size:.82rem;color:#6b8e6b'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
                    st.progress(min(prob,1.0))
            with col_c:
                fig2=fig_disease_bars(final_probs)
                if fig2: st.pyplot(fig2, transparent=True); plt.close(fig2)

        if settings["show_features"]:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### 🔬 Quantitative Leaf Feature Analysis")
            cf1,cf2=st.columns(2); items=list(features.items()); half=len(items)//2
            with cf1:
                for k,v in items[:half]: st.markdown(f'<div class="feat-row"><span class="feat-key">{k}</span><span class="feat-val">{v}</span></div>', unsafe_allow_html=True)
            with cf2:
                for k,v in items[half:]: st.markdown(f'<div class="feat-row"><span class="feat-key">{k}</span><span class="feat-val">{v}</span></div>', unsafe_allow_html=True)

        if settings["show_training"]:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### 📉 Model Training History")
            fig3=fig_training_curves()
            if fig3: st.pyplot(fig3, use_container_width=True); plt.close(fig3)
            else: st.info("training_history.json not found.")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### 💊 AI Recommendations")
        gemini_treatment  = gemini_parsed.get("treatment","")
        gemini_prevention = gemini_parsed.get("prevention","")
        gemini_symptoms   = gemini_parsed.get("symptoms","")
        tab1,tab2,tab3,tab4=st.tabs(["🧪 Treatment","✂️ Pruning","🛡️ Prevention","🤖 Canopy Advice"])
        with tab1:
            for item in suggestions["treatment"]: st.markdown(f'<div class="suggestion-item">• {item}</div>', unsafe_allow_html=True)
        with tab2:
            for item in suggestions["pruning"]:   st.markdown(f'<div class="suggestion-item">• {item}</div>', unsafe_allow_html=True)
        with tab3:
            for item in suggestions["prevention"]: st.markdown(f'<div class="suggestion-item">• {item}</div>', unsafe_allow_html=True)
        with tab4:
            if gemini_symptoms:   st.markdown("**Symptoms:**");   st.markdown(f'<div class="suggestion-item">{gemini_symptoms}</div>',   unsafe_allow_html=True)
            if gemini_treatment:  st.markdown("**Treatment:**");  st.markdown(f'<div class="suggestion-item">{gemini_treatment}</div>',  unsafe_allow_html=True)
            if gemini_prevention: st.markdown("**Prevention:**"); st.markdown(f'<div class="suggestion-item">{gemini_prevention}</div>', unsafe_allow_html=True)
            if not any([gemini_symptoms,gemini_treatment,gemini_prevention]): st.info("No structured Canopy advice for this image.")

        with st.expander("📐 Model Architecture Details"):
            cm1,cm2=st.columns(2)
            with cm1: st.markdown("**GrapeCanopyNet (U-Net)**"); st.markdown(UNET_TABLE, unsafe_allow_html=True)
            with cm2: st.markdown("**GrapeLeafCNN**"); st.markdown(CNN_TABLE, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        lines=["GRAPEAI CROP INTELLIGENCE REPORT","="*56,
               f"Image        : {uploaded_file.name}",
               f"Image Type   : {img_type['type'].capitalize()} ({img_type['confidence']:.0%})",
               f"Engine Source: {badge_label}",f"Engine Reason: {reason}","",
               "CANOPY ANALYSIS",f"  Coverage: {canopy['coverage_percent']:.2f}%",
               f"  Status  : {c_info['label']}","",
               f"DISEASE DETECTION ({badge_label})",f"  Condition  : {final_disease}",
               f"  Confidence : {final_conf*100:.1f}%",f"  Alert      : {final_urgency}","",
               f"  CNN result   : {cnn_display} ({cnn_conf*100:.1f}%)",
               f"  Canopy AI result: {gemini_parsed.get('disease_name') or 'N/A'}","",
               "SEVERITY",f"  {final_severity['severity_pct']}% — {final_severity['severity_label']}","",
               "LEAF FEATURES"]
        for k,v in features.items(): lines.append(f"  {k:<28}: {v}")
        lines+=["","CLASS PROBABILITIES"]
        for n,p in final_probs.items(): lines.append(f"  {n:<40}: {p*100:.1f}%")
        lines+=["","TREATMENT"]
        for t in suggestions["treatment"]: lines.append(f"  * {t}")
        lines+=["","PRUNING"]
        for p in suggestions["pruning"]: lines.append(f"  * {p}")
        lines+=["","PREVENTION"]
        for p in suggestions["prevention"]: lines.append(f"  * {p}")
        if gemini_raw: lines+=["","CANOPY EXPERT ANALYSIS","-"*40,gemini_raw.strip()]
        lines+=["","="*56,"GrapeAI Intelligence System — B.Tech Final Year Project"]
        st.download_button("⬇️  Download Full Report", data="\n".join(lines),
                           file_name=f"grapeai_{uploaded_file.name.split('.')[0]}.txt", mime="text/plain")

    st.markdown('<div class="footer">GrapeAI Intelligence System &middot; GrapeCanopyNet + GrapeLeafCNN + Canopy AI &middot; Unified Disease Decision Engine &middot; B.Tech Final Year Project</div>', unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_page()
else:
    page, settings = render_sidebar()
    page_key = page.split(" ", 1)[-1].strip()
    if   page_key == "Dashboard":      page_dashboard()
    elif page_key == "Analysis":       page_analysis(settings)
    elif page_key == "Model Registry": page_model_registry()
    elif page_key == "Dataset Info":   page_dataset()
    elif page_key == "About":          page_about()