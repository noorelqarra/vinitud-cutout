import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import unicodedata
import re
import streamlit as st
import io

CANVAS_SIZE = 2000
TARGET_RATIO = 0.72

OUTPUT_DIR = Path.home() / "VinitudCutout_Output"
OUTPUT_DIR.mkdir(exist_ok=True)

def slugify(text: str) -> str:
    text = (text or "").strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text if text else "product"

def remove_white_background(img: Image.Image, threshold: int, softness: int) -> Image.Image:
    img_np = np.array(img.convert("RGBA"))
    rgb = img_np[..., :3]
    alpha = img_np[..., 3]

    # consider background as near-white
    white_mask = np.all(rgb > threshold, axis=2)

    alpha_new = alpha.copy()
    alpha_new[white_mask] = 0

    # soften edges (helps halos and light shadows)
    blur = cv2.GaussianBlur(alpha_new, (0, 0), softness)
    img_np[..., 3] = blur

    return Image.fromarray(img_np)

def bbox_from_alpha(img: Image.Image):
    alpha = np.array(img)[..., 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

def normalize_canvas(img: Image.Image) -> Image.Image:
    bbox = bbox_from_alpha(img)
    if not bbox:
        return img

    x1, y1, x2, y2 = bbox
    crop = img.crop((x1, y1, x2 + 1, y2 + 1))

    w, h = crop.size
    target_h = int(CANVAS_SIZE * TARGET_RATIO)
    scale = target_h / h

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    left = (CANVAS_SIZE - new_w) // 2
    top = (CANVAS_SIZE - new_h) // 2
    canvas.paste(resized, (left, top), resized)
    return canvas

def process(img: Image.Image, brand: str, product: str, year: str, threshold: int, softness: int):
    cutout = remove_white_background(img, threshold, softness)
    final = normalize_canvas(cutout)

    filename = f"{slugify(brand)}-{slugify(product)}-{slugify(year)}-transparent-2000.png"
    save_path = OUTPUT_DIR / filename
    final.save(save_path, "PNG")

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)
    return final, filename, save_path, buf

st.set_page_config(page_title="Vinitud Cutout", layout="centered")
st.title("🍾 Vinitud Cutout Tool")
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia al 72% (locale, gratuito).")

uploaded = st.file_uploader("Carica un'immagine (sfondo bianco)", type=["png", "jpg", "jpeg"])

col1, col2, col3 = st.columns(3)
with col1:
    brand = st.text_input("Brand", "")
with col2:
    product = st.text_input("Prodotto", "")
with col3:
    year = st.text_input("Anno", "")

threshold = st.slider("Soglia bianco", 200, 255, 240, 1)
softness = st.slider("Morbidezza bordo", 1, 25, 8, 1)

if uploaded:
    img = Image.open(uploaded).convert("RGBA")
    st.image(img, caption="Input", use_container_width=True)

    if st.button("Processa"):
        final, filename, save_path, buf = process(img, brand, product, year, threshold, softness)
        st.success(f"Fatto! Salvato in: {save_path}")
        st.image(final, caption="Output (trasparente)", use_container_width=True)

        st.download_button(
            label="Scarica PNG",
            data=buf,
            file_name=filename,
            mime="image/png",
        )
else:
    st.info("Carica una foto per iniziare.")
