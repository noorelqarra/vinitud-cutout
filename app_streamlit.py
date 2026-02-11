import io
import re
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter


# ----------------------------
# Config
# ----------------------------
CANVAS_SIZE = 2000
BOTTLE_HEIGHT_RATIO = 0.72  # 72% come da tua specifica
OUTPUT_DIR_NAME = "VinitudCutout_Output"


def safe_slug(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "output"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "", text)
    return text or "output"


def cutout_white_background(
    img: Image.Image,
    thr_white: int,
    feather: int,
) -> Image.Image:
    """
    Rimuove sfondo bianco/near-white creando alpha.
    Feather applica una sfocatura al bordo.
    """
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.uint8)

    rgb = arr[:, :, :3]
    # Pixel considerati "bianchi" se tutti i canali >= soglia
    white_mask = (rgb[:, :, 0] >= thr_white) & (rgb[:, :, 1] >= thr_white) & (rgb[:, :, 2] >= thr_white)

    # Alpha: 0 dove è bianco, 255 dove è oggetto
    alpha = np.where(white_mask, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")

    if feather and feather > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather))

    # Applica alpha all'immagine
    out = img.copy()
    out.putalpha(alpha_img)
    return out


def trim_transparent(img_rgba: Image.Image, alpha_threshold: int = 5) -> Image.Image:
    """
    Croppa al bounding box dei pixel con alpha > alpha_threshold.
    """
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return img_rgba
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return img_rgba.crop((x0, y0, x1 + 1, y1 + 1))


def place_on_canvas(img_rgba: Image.Image, canvas_size: int, height_ratio: float) -> Image.Image:
    """
    Ridimensiona l'oggetto per avere altezza = height_ratio del canvas, e lo centra su canvas trasparente.
    """
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    w, h = img_rgba.size
    target_h = int(canvas_size * height_ratio)
    scale = target_h / max(h, 1)
    target_w = int(w * scale)

    resized = img_rgba.resize((target_w, target_h), Image.LANCZOS)

    # Center
    x = (canvas_size - target_w) // 2
    y = (canvas_size - target_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def to_png_bytes(img_rgba: Image.Image) -> bytes:
    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def try_save_local(png_bytes: bytes, filename: str) -> Path | None:
    """
    Salva in ~/VinitudCutout_Output solo se possibile (locale).
    Su Streamlit Cloud di solito non ti serve perché scarichi col bottone.
    """
    try:
        out_dir = Path.home() / OUTPUT_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_bytes(png_bytes)
        return out_path
    except Exception:
        return None


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Vinitud Cutout Tool", page_icon="🍾", layout="wide")

st.title("🍾 Vinitud Cutout Tool")
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia al 72% (locale o cloud).")

uploaded = st.file_uploader("Carica un'immagine (sfondo bianco)", type=["png", "jpg", "jpeg"])

col1, col2, col3 = st.columns(3)
with col1:
    brand = st.text_input("Brand")
with col2:
    prodotto = st.text_input("Prodotto")
with col3:
    anno = st.text_input("Anno")

thr_white = st.slider("Soglia bianco", 0, 255, 240)
feather = st.slider("Morbidezza bordo", 0, 30, 8)

run = st.button("Genera output", type="primary", disabled=(uploaded is None))

# Stato per mantenere output anche se muovi slider dopo
if "last_png" not in st.session_state:
    st.session_state.last_png = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None
if "last_saved_path" not in st.session_state:
    st.session_state.last_saved_path = None

if uploaded is None:
    st.info("Carica una foto per iniziare.")
else:
    # Preview input
    try:
        input_img = Image.open(uploaded).convert("RGBA")
        st.image(input_img, caption="Input", width=420)
    except Exception as e:
        st.error(f"Errore nel leggere l'immagine: {e}")
        st.stop()

    if run:
        # 1) Cutout
        cut = cutout_white_background(input_img, thr_white=thr_white, feather=feather)

        # 2) Crop su oggetto
        cut = trim_transparent(cut, alpha_threshold=5)

        # 3) Canvas 2000x2000 con scaling 72%
        final_img = place_on_canvas(cut, canvas_size=CANVAS_SIZE, height_ratio=BOTTLE_HEIGHT_RATIO)

        # 4) Bytes + file name
        fname = f"{safe_slug(brand)}-{safe_slug(prodotto)}-{safe_slug(anno)}-transparent-2000.png"
        png_bytes = to_png_bytes(final_img)

        # 5) salva se possibile (locale)
        saved_path = try_save_local(png_bytes, fname)

        st.session_state.last_png = png_bytes
        st.session_state.last_filename = fname
        st.session_state.last_saved_path = saved_path

        st.success("Output generato ✅")

    # Se esiste un output in sessione, mostrali sempre (non spariscono)
    if st.session_state.last_png:
        st.subheader("Output")

        out_img = Image.open(io.BytesIO(st.session_state.last_png)).convert("RGBA")
        st.image(out_img, caption="PNG trasparente 2000×2000", width=520)

        st.download_button(
            label="Scarica PNG",
            data=st.session_state.last_png,
            file_name=st.session_state.last_filename or "output.png",
            mime="image/png",
        )

        if st.session_state.last_saved_path:
            st.caption(f"Salvato in locale: {st.session_state.last_saved_path}")
        else:
            st.caption("Nota: su Streamlit Cloud non salvi sul tuo Mac. Usa il bottone “Scarica PNG”.")
