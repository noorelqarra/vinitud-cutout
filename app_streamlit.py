import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import datetime
import io

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="Vinitud Cutout Tool",
    layout="wide",
)

# =============================
# TITLE
# =============================

st.markdown("""
# 🍾 Vinitud Cutout Tool
Sondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (cloud / locale).
""")

# =============================
# OUTPUT DIR
# =============================

OUTPUT_DIR = Path.home() / "VinitudCutout_Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# LAYOUT
# =============================

left, right = st.columns([1, 2.4])

# =============================
# LEFT PANEL — INPUT
# =============================

with left:

    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"]
    )

    brand = st.text_input("Brand")
    prodotto = st.text_input("Prodotto")
    anno = st.text_input("Anno")

    st.markdown("---")

    soglia = st.slider("Soglia bianco", 180, 255, 218)
    blur = st.slider("Morbidezza bordo", 0, 20, 3)

    st.markdown("")

    generate_btn = st.button("🚀 Genera output", use_container_width=True)

    st.markdown("---")

    if uploaded_file:
        st.caption("Preview input")
        input_img = Image.open(uploaded_file).convert("RGB")
        st.image(input_img, width=260)

# =============================
# PROCESSING
# =============================

result_img = None
filename = None

if uploaded_file and generate_btn:

    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    gray = np.mean(img_np, axis=2)
    mask = gray < soglia

    alpha = np.zeros_like(gray, dtype=np.uint8)
    alpha[mask] = 255

    alpha_img = Image.fromarray(alpha)

    if blur > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur))

    rgba = img.copy()
    rgba.putalpha(alpha_img)

    canvas = Image.new("RGBA", (2000, 2000), (0, 0, 0, 0))

    scale = 0.72
    new_h = int(2000 * scale)
    ratio = new_h / rgba.height
    new_w = int(rgba.width * ratio)

    resized = rgba.resize((new_w, new_h), Image.LANCZOS)

    pos = ((2000 - new_w) // 2, (2000 - new_h) // 2)
    canvas.paste(resized, pos, resized)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    name_parts = [brand, prodotto, anno]
    name = "_".join([p for p in name_parts if p.strip()])

    if not name:
        name = "vinitud"

    filename = f"{name}_{ts}.png"

    output_path = OUTPUT_DIR / filename
    canvas.save(output_path)

    result_img = canvas

# =============================
# RIGHT PANEL — OUTPUT
# =============================

with right:

    st.subheader("Output")

    if result_img:

        st.image(result_img, width=420)

        st.caption("PNG trasparente 2000×2000")

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            "⬇ Scarica PNG",
            byte_im,
            file_name=filename,
            mime="image/png"
        )

        st.success(f"Salvato anche in locale: {OUTPUT_DIR}/{filename}")

    else:
        st.info("Carica un'immagine e premi **Genera output**.")
