import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

st.set_page_config(page_title="Vinitud Cutout Tool", layout="centered")

st.title("🍾 Vinitud Cutout Tool")
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia al 72% (locale, gratuito).")

uploaded = st.file_uploader(
    "Carica un'immagine (sfondo bianco)",
    type=["png", "jpg", "jpeg"]
)

col1, col2, col3 = st.columns(3)
brand = col1.text_input("Brand")
prodotto = col2.text_input("Prodotto")
anno = col3.text_input("Anno")

threshold = st.slider("Soglia bianco", 200, 255, 240)
soft = st.slider("Morbidezza bordo", 0, 20, 8)

OUTPUT_DIR = Path.home() / "VinitudCutout_Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cutout_white_bg_pillow(img: Image.Image, thr: int, soften: int) -> Image.Image:
    # Ensure RGB
    img = img.convert("RGB")
    arr = np.array(img).astype(np.uint8)

    # White mask: pixel considered background if ALL channels >= threshold
    bg = (arr[:, :, 0] >= thr) & (arr[:, :, 1] >= thr) & (arr[:, :, 2] >= thr)

    # Build alpha: 0 for background, 255 for subject
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")

    # Soften edges if requested
    if soften > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=soften))

    rgba = img.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba

def place_on_canvas(subject_rgba: Image.Image, canvas_size=2000, scale=0.72) -> Image.Image:
    # Crop to non-transparent bbox to avoid huge empty margins
    bbox = subject_rgba.getbbox()
    if bbox:
        subject_rgba = subject_rgba.crop(bbox)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    # Fit subject into target box (scale of canvas)
    target = int(canvas_size * scale)
    w, h = subject_rgba.size
    if w == 0 or h == 0:
        return canvas

    ratio = min(target / w, target / h)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    subject_resized = subject_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on canvas
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2
    canvas.alpha_composite(subject_resized, (x, y))
    return canvas

if not uploaded:
    st.info("Carica una foto per iniziare.")
    st.stop()

img_in = Image.open(uploaded)

# Preview input
st.image(img_in, caption="Input", use_container_width=True)

# Process
subject = cutout_white_bg_pillow(img_in, threshold, soft)
final_img = place_on_canvas(subject, canvas_size=2000, scale=0.72)

st.image(final_img, caption="Output (PNG trasparente 2000×2000)", use_container_width=True)

def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "-").replace("\\", "-")
    return "".join(ch for ch in s if ch.isalnum() or ch in " -_").strip().replace(" ", "_")

filename_parts = [safe_name(brand), safe_name(prodotto), safe_name(anno)]
filename_parts = [p for p in filename_parts if p]
out_name = "vinitud-cutout.png" if not filename_parts else "-".join(filename_parts) + "-transparent-2000.png"
out_path = OUTPUT_DIR / out_name

# Save + download
final_img.save(out_path, format="PNG")

with open(out_path, "rb") as f:
    st.download_button(
        "⬇️ Scarica PNG",
        data=f,
        file_name=out_name,
        mime="image/png"
    )

st.caption(f"Salvato anche in: {out_path}")
