import re
from collections import deque
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


def _flood_fill_background(bg_candidate: np.ndarray) -> np.ndarray:
    """
    bg_candidate: boolean mask True where pixel is "near white"
    Returns: boolean mask True where pixel is REAL background (near-white AND connected to borders)
    """
    h, w = bg_candidate.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    # enqueue border pixels that are candidates
    def push(y, x):
        if 0 <= y < h and 0 <= x < w and (not visited[y, x]) and bg_candidate[y, x]:
            visited[y, x] = True
            q.append((y, x))

    # top & bottom rows
    for x in range(w):
        push(0, x)
        push(h - 1, x)
    # left & right cols
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    # BFS 4-neighborhood
    while q:
        y, x = q.popleft()
        if y > 0: push(y - 1, x)
        if y < h - 1: push(y + 1, x)
        if x > 0: push(y, x - 1)
        if x < w - 1: push(y, x + 1)

    # visited == background connected to edges
    return visited


def remove_white_background_rgba(
    img_rgb: Image.Image,
    white_threshold: int,
    feather_px: int,
    downscale_max: int = 900,
) -> Image.Image:
    """
    Robust background removal:
    1) detect near-white pixels
    2) mark as background only those near-white pixels connected to image borders
    This prevents eating white labels/reflections on the bottle.
    """
    img_rgb = img_rgb.convert("RGB")
    w0, h0 = img_rgb.size

    # downscale for flood-fill speed (then upscale mask)
    scale = min(1.0, downscale_max / max(w0, h0))
    if scale < 1.0:
        w1, h1 = int(w0 * scale), int(h0 * scale)
        small = img_rgb.resize((w1, h1), Image.Resampling.BILINEAR)
    else:
        small = img_rgb
        w1, h1 = w0, h0

    arr = np.asarray(small).astype(np.uint8)

    # "near white" candidate: all channels above threshold
    bg_candidate = (arr[:, :, 0] > white_threshold) & (arr[:, :, 1] > white_threshold) & (arr[:, :, 2] > white_threshold)

    # flood fill from borders to keep only real background
    bg_real_small = _flood_fill_background(bg_candidate)

    # convert to alpha: background -> 0, else -> 255
    alpha_small = np.where(bg_real_small, 0, 255).astype(np.uint8)
    alpha_img_small = Image.fromarray(alpha_small, mode="L")

    # upscale alpha mask back to original size
    if (w1, h1) != (w0, h0):
        alpha_img = alpha_img_small.resize((w0, h0), Image.Resampling.BILINEAR)
    else:
        alpha_img = alpha_img_small

    # feather edge
    if feather_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=int(feather_px)))

    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba


def compose_on_canvas(
    rgba: Image.Image,
    canvas_size: int = 2000,
    target_height_ratio: float = 0.72,
) -> Image.Image:
    """
    NO autocrop: keep original proportions, scale by height ratio, center on 2000x2000 transparent canvas.
    """
    w, h = rgba.size
    target_h = int(canvas_size * target_height_ratio)

    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (fix etichette chiare).")

if "out_png" not in st.session_state:
    st.session_state.out_png = None
if "out_filename" not in st.session_state:
    st.session_state.out_filename = None
if "out_path" not in st.session_state:
    st.session_state.out_path = None

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Input")

    uploaded = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"],
    )

    brand = st.text_input("Brand")
    prodotto = st.text_input("Prodotto")
    anno = st.text_input("Anno")

    white_threshold = st.slider("Soglia bianco", 200, 255, 232, 1)
    feather_px = st.slider("Morbidezza bordo", 0, 20, 3, 1)
    target_ratio = st.slider("Dimensione bottiglia (altezza su 2000px)", 0.55, 0.85, 0.72, 0.01)

    st.markdown("---")
    generate = st.button("Genera output", type="primary", use_container_width=True)

    st.markdown("**Preview input**")
    if uploaded:
        img_preview = Image.open(uploaded).convert("RGB")
        thumb = img_preview.copy()
        thumb.thumbnail((520, 520))
        st.image(thumb, width=320)
    else:
        st.info("Carica una foto per iniziare.")

with right:
    st.subheader("Output")

    if generate:
        if uploaded is None:
            st.error("Carica un'immagine prima.")
        else:
            img_in = Image.open(uploaded).convert("RGB")

            rgba = remove_white_background_rgba(
                img_rgb=img_in,
                white_threshold=int(white_threshold),
                feather_px=int(feather_px),
                downscale_max=900,
            )

            canvas = compose_on_canvas(
                rgba=rgba,
                canvas_size=2000,
                target_height_ratio=float(target_ratio),
            )

            filename = build_filename(brand, prodotto, anno)
            png_bytes = pil_to_png_bytes(canvas)

            out_dir = Path.home() / "VinitudCutout_Output"
            out_dir.mkdir(exist_ok=True)

            out_path = out_dir / filename
            out_path.write_bytes(png_bytes)  # sovrascrive come richiesto

            st.session_state.out_png = png_bytes
            st.session_state.out_filename = filename
            st.session_state.out_path = str(out_path)

    if st.session_state.out_png:
        out_img = Image.open(BytesIO(st.session_state.out_png)).convert("RGBA")

        st.image(out_img, width=520)
        st.caption("PNG trasparente 2000×2000")

        st.download_button(
            "Scarica PNG",
            data=st.session_state.out_png,
            file_name=st.session_state.out_filename,
            mime="image/png",
        )

        st.caption(f"Salvato anche in locale: `{st.session_state.out_path}`")
        st.caption("Nota: su Streamlit Cloud il salvataggio su disco è temporaneo. Conta il download.")
    else:
        st.info("Genera un output per vederlo qui.")
