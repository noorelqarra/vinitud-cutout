# app_streamlit.py
# Vinitud Cutout Tool — white background -> transparent PNG 2000x2000, bottle ~72%
# NO OpenCV (cv2). Uses only Pillow + numpy so it works on Streamlit Cloud.

import io
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter


# ----------------------------
# Helpers
# ----------------------------
def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "vinitud"


def remove_white_bg_rgba(img_rgba: Image.Image, threshold: int, feather: int) -> Image.Image:
    """
    Build alpha mask: pixels close to white become transparent.
    threshold: 0..255 (higher => removes more)
    feather:   0..30 approx (soft edge)
    """
    arr = np.array(img_rgba).astype(np.uint8)  # (H,W,4)
    rgb = arr[..., :3].astype(np.int16)

    # "whiteness" as minimum channel value (conservative): if min channel is high, it's white-ish
    whiteness = np.min(rgb, axis=2)  # 0..255

    # mask_foreground: 1 where NOT white (keep), 0 where white (remove)
    mask_fg = (whiteness < threshold).astype(np.uint8) * 255  # 0/255

    mask_img = Image.fromarray(mask_fg, mode="L")

    if feather and feather > 0:
        # blur the mask edges for smoother transparency
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather)))

    out = img_rgba.copy()
    out.putalpha(mask_img)
    return out


def compose_2000_square(subject_rgba: Image.Image, target_size: int = 2000, subject_height_ratio: float = 0.72) -> Image.Image:
    """
    Centers the subject on a transparent 2000x2000 canvas, scaling it
    so its height is ~72% of the canvas.
    """
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    # bounding box of non-transparent pixels
    alpha = np.array(subject_rgba.getchannel("A"))
    ys, xs = np.where(alpha > 5)
    if len(xs) == 0 or len(ys) == 0:
        return canvas  # nothing detected

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    cropped = subject_rgba.crop((left, top, right + 1, bottom + 1))

    # scale to target height ratio
    target_h = int(target_size * subject_height_ratio)
    w, h = cropped.size
    if h <= 0:
        return canvas

    scale = target_h / h
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def try_save_local(png_bytes: bytes, filename: str) -> str | None:
    """
    Optional: if running locally, save a copy to ~/VinitudCutout_Output.
    On Streamlit Cloud it may fail (read-only). We just ignore failures.
    """
    try:
        out_dir = Path.home() / "VinitudCutout_Output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_bytes(png_bytes)
        return str(out_path)
    except Exception:
        return None


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Vinitud Cutout Tool", layout="wide")

st.title("Vinitud Cutout Tool")
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (cloud / locale).")

# Keep output stable until user clicks "Genera output"
if "last_output_bytes" not in st.session_state:
    st.session_state.last_output_bytes = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None
if "last_saved_path" not in st.session_state:
    st.session_state.last_saved_path = None

# Layout: LEFT narrow input, RIGHT wide output
col_left, col_right = st.columns([1, 2.2], gap="large")

with col_left:
    st.subheader("Input")

    uploaded = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    brand = st.text_input("Brand", value="")
    prodotto = st.text_input("Prodotto", value="")
    anno = st.text_input("Anno", value="")

    threshold = st.slider("Soglia bianco", min_value=200, max_value=255, value=214, step=1, help="Più alto = rimuove più bianco.")
    feather = st.slider("Morbidezza bordo", min_value=0, max_value=15, value=3, step=1, help="Sfuma leggermente il bordo della trasparenza.")

    st.divider()

    # Preview input (under controls)
    if uploaded:
        try:
            input_img = Image.open(uploaded).convert("RGBA")
            st.caption("Preview input")
            st.image(input_img, width=260)
        except Exception as e:
            st.error(f"Impossibile leggere l'immagine: {e}")
            input_img = None
    else:
        input_img = None
        st.info("Carica una foto per iniziare.")

    generate = st.button("Genera output", type="primary", use_container_width=True, disabled=(input_img is None))

with col_right:
    st.subheader("Output")

    if generate and input_img is not None:
        # 1) Remove white bg
        cut = remove_white_bg_rgba(input_img, threshold=threshold, feather=feather)

        # 2) Compose 2000x2000 with ~72% subject
        out_img = compose_2000_square(cut, target_size=2000, subject_height_ratio=0.72)

        # 3) Build filename
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{slugify(brand)}_{slugify(prodotto)}_{slugify(anno)}_{stamp}.png"

        # 4) PNG bytes
        buf = io.BytesIO()
        out_img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

        # 5) Store in session (so it stays visible if user changes sliders)
        st.session_state.last_output_bytes = png_bytes
        st.session_state.last_filename = fname
        st.session_state.last_saved_path = try_save_local(png_bytes, fname)

    # Render last output if present
    if st.session_state.last_output_bytes:
        out_preview = Image.open(io.BytesIO(st.session_state.last_output_bytes)).convert("RGBA")

        # Output preview size (not giant)
        st.image(out_preview, width=520)

        st.caption("PNG trasparente 2000×2000")

        st.download_button(
            label="Scarica PNG",
            data=st.session_state.last_output_bytes,
            file_name=st.session_state.last_filename or "vinitud_output.png",
            mime="image/png",
        )

        if st.session_state.last_saved_path:
            st.caption(f"Salvato anche in locale: {st.session_state.last_saved_path}")
        else:
            st.caption("Nota: su Streamlit Cloud il file non viene salvato su disco (scaricalo con il pulsante).")
    else:
        st.info("Carica un’immagine e clicca **Genera output**.")
