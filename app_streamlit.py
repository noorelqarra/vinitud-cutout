import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import streamlit as st


# -------------------------
# Helpers
# -------------------------

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = (
        s.replace("à", "a").replace("è", "e").replace("é", "e")
        .replace("ì", "i").replace("ò", "o").replace("ù", "u")
    )
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def build_filename(brand: str, prodotto: str, anno: str) -> str:
    b = slugify(brand)
    p = slugify(prodotto)
    a = slugify(str(anno))
    parts = [x for x in [b, p, a] if x]
    if not parts:
        parts = ["output"]
    return "_".join(parts) + ".png"


def remove_white_background_rgba(
    img_rgb: Image.Image,
    white_threshold: int,
    feather_px: int,
) -> Image.Image:

    img_rgb = img_rgb.convert("RGB")
    arr = np.asarray(img_rgb).astype(np.uint8)

    bg = (
        (arr[:, :, 0] > white_threshold)
        & (arr[:, :, 1] > white_threshold)
        & (arr[:, :, 2] > white_threshold)
    )

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")

    if feather_px > 0:
        alpha_img = alpha_img.filter(
            ImageFilter.GaussianBlur(radius=int(feather_px))
        )

    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba


def compose_on_canvas(
    rgba: Image.Image,
    canvas_size: int = 2000,
    target_height_ratio: float = 0.72,
) -> Image.Image:

    w, h = rgba.size
    target_h = int(canvas_size * target_height_ratio)

    scale = target_h / float(h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = rgba.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2

    canvas.paste(resized, (x, y), resized)
    return canvas


def pil_to_png_bytes(img: Image.Image) -> bytes:
    bio = BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Vinitud Cutout Tool", layout="wide")

st.title("🍾 Vinitud Cutout Tool")
st.caption(
    "Sfondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (cloud / locale)."
)

if "out_png" not in st.session_state:
    st.session_state.out_png = None
if "out_filename" not in st.session_state:
    st.session_state.out_filename = None
if "out_path" not in st.session_state:
    st.session_state.out_path = None


left, right = st.columns([1, 2], gap="large")

# -------------------------
# LEFT – INPUT
# -------------------------

with left:
    st.subheader("Input")

    uploaded = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"],
    )

    brand = st.text_input("Brand")
    prodotto = st.text_input("Prodotto")
    anno = st.text_input("Anno")

    white_threshold = st.slider(
        "Soglia bianco", 200, 255, 218, 1
    )

    feather_px = st.slider(
        "Morbidezza bordo", 0, 20, 3, 1
    )

    target_ratio = st.slider(
        "Dimensione bottiglia (altezza su 2000px)",
        0.55,
        0.85,
        0.72,
        0.01,
    )

    st.markdown("---")

    generate = st.button(
        "Genera output",
        type="primary",
        use_container_width=True,
    )

    st.markdown("**Preview input**")

    if uploaded:
        img_preview = Image.open(uploaded).convert("RGB")
        thumb = img_preview.copy()
        thumb.thumbnail((520, 520))
        st.image(thumb, width=320)
    else:
        st.info("Carica una foto per iniziare.")


# -------------------------
# RIGHT – OUTPUT
# -------------------------

with right:
    st.subheader("Output")

    if generate:
        if uploaded is None:
            st.error("Carica un'immagine prima.")
        else:
            img_in = Image.open(uploaded).convert("RGB")

            rgba = remove_white_background_rgba(
                img_in,
                white_threshold,
                feather_px,
            )

            canvas = compose_on_canvas(
                rgba,
                canvas_size=2000,
                target_height_ratio=target_ratio,
            )

            filename = build_filename(
                brand, prodotto, anno
            )

            png_bytes = pil_to_png_bytes(canvas)

            out_dir = Path.home() / "VinitudCutout_Output"
            out_dir.mkdir(exist_ok=True)

            out_path = out_dir / filename
            out_path.write_bytes(png_bytes)

            st.session_state.out_png = png_bytes
            st.session_state.out_filename = filename
            st.session_state.out_path = str(out_path)

    if st.session_state.out_png:
        out_img = Image.open(
            BytesIO(st.session_state.out_png)
        )

        st.image(out_img, width=520)

        st.caption("PNG trasparente 2000×2000")

        st.download_button(
            "Scarica PNG",
            data=st.session_state.out_png,
            file_name=st.session_state.out_filename,
            mime="image/png",
        )

        st.caption(
            f"Salvato anche in locale: `{st.session_state.out_path}`"
        )

    else:
        st.info("Genera un output per vederlo qui.")
