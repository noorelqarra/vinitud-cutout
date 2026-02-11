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
    s = re.sub(r"[^a-z0-9]+", "_", s)   # non alfanumerico -> _
    s = re.sub(r"_+", "_", s)          # _ ripetuti -> singolo
    return s.strip("_")


def build_filename(brand: str, prodotto: str, anno: str) -> str:
    b = slugify(brand)
    p = slugify(prodotto)
    a = slugify(str(anno)) if anno is not None else ""
    parts = [x for x in [b, p, a] if x]
    if not parts:
        parts = ["output"]
    return "_".join(parts) + ".png"


def remove_white_background_rgba(img_rgb: Image.Image, white_threshold: int, feather_px: int) -> Image.Image:
    """
    Crea alpha=0 dove il pixel è "quasi bianco" (tutti i canali > soglia).
    Feather: blur della maschera (bordi più morbidi).
    """
    img_rgb = img_rgb.convert("RGB")
    arr = np.asarray(img_rgb).astype(np.uint8)

    # mask True = background (quasi bianco)
    bg = (arr[:, :, 0] > white_threshold) & (arr[:, :, 1] > white_threshold) & (arr[:, :, 2] > white_threshold)

    # alpha: 0 sul bg, 255 sul soggetto
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")

    if feather_px and feather_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=int(feather_px)))

    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba


def bbox_from_alpha(rgba: Image.Image, min_alpha: int = 10):
    a = np.asarray(rgba.split()[-1])
    ys, xs = np.where(a >= min_alpha)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # PIL crop box è (left, upper, right, lower) con right/lower esclusivi
    return (int(x0), int(y0), int(x1 + 1), int(y1 + 1))


def compose_on_canvas(
    rgba: Image.Image,
    canvas_size: int = 2000,
    target_height_ratio: float = 0.72,
    autocrop: bool = True,
) -> Image.Image:
    """
    1) (opzionale) crop sul bbox del soggetto
    2) resize per avere altezza soggetto = target_height_ratio * canvas
    3) incolla al centro su canvas trasparente
    """
    bbox = bbox_from_alpha(rgba)
    work = rgba

    if autocrop and bbox is not None:
        work = rgba.crop(bbox)

    # se bbox non c'è (edge case), usa tutta l'immagine
    w, h = work.size
    if h <= 0 or w <= 0:
        # fallback: canvas vuota
        return Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    target_h = int(canvas_size * target_height_ratio)
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    work_resized = work.resize((new_w, new_h), resample=Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2
    canvas.paste(work_resized, (x, y), work_resized)
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
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (cloud / locale).")

# Session state per output stabile
if "out_png" not in st.session_state:
    st.session_state.out_png = None
if "out_filename" not in st.session_state:
    st.session_state.out_filename = None
if "out_path" not in st.session_state:
    st.session_state.out_path = None
if "input_preview" not in st.session_state:
    st.session_state.input_preview = None

# Layout: sinistra stretta, destra larga
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Input")

    uploaded = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False
    )

    brand = st.text_input("Brand", value="")
    prodotto = st.text_input("Prodotto", value="")
    anno = st.text_input("Anno", value="")

    white_threshold = st.slider("Soglia bianco", min_value=200, max_value=255, value=218, step=1)
    feather_px = st.slider("Morbidezza bordo", min_value=0, max_value=20, value=3, step=1)

    st.markdown("---")
    autocrop = st.checkbox("Auto-crop bottiglia (consigliato)", value=True)
    target_ratio = st.slider("Dimensione bottiglia (altezza su 2000px)", 0.55, 0.85, 0.72, 0.01)

    st.markdown("---")

    # Pulsante sempre visibile
    generate = st.button("Genera output", type="primary", use_container_width=True)

    # Preview input sotto i controlli (piccolo)
    st.markdown("**Preview input**")
    if uploaded is not None:
        try:
            img_in = Image.open(uploaded).convert("RGB")
            st.session_state.input_preview = img_in.copy()
            # preview piccola
            preview = img_in.copy()
            preview.thumbnail((520, 520))
            st.image(preview, width=320)
        except Exception as e:
            st.error(f"Errore nel leggere l'immagine: {e}")
    else:
        st.info("Carica una foto per iniziare.")


with right:
    st.subheader("Output")

    # Quando clicchi genera
    if generate:
        if uploaded is None:
            st.error("Carica un'immagine prima di generare l’output.")
        else:
            try:
                img_in = Image.open(uploaded).convert("RGB")

                rgba = remove_white_background_rgba(
                    img_rgb=img_in,
                    white_threshold=int(white_threshold),
                    feather_px=int(feather_px),
                )

                canvas = compose_on_canvas(
                    rgba=rgba,
                    canvas_size=2000,
                    target_height_ratio=float(target_ratio),
                    autocrop=bool(autocrop),
                )

                filename = build_filename(brand, prodotto, anno)
                out_bytes = pil_to_png_bytes(canvas)

                # Salva in locale (vale sia locale sia cloud; nel cloud è filesystem temporaneo)
                out_dir = Path.home() / "VinitudCutout_Output"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / filename
                out_path.write_bytes(out_bytes)  # sovrascrive (come richiesto)

                st.session_state.out_png = out_bytes
                st.session_state.out_filename = filename
                st.session_state.out_path = str(out_path)

            except Exception as e:
                st.error(f"Errore durante la generazione: {e}")

    # Se ho un output in sessione, lo mostro sempre (anche se muovi slider)
    if st.session_state.out_png:
        out_img = Image.open(BytesIO(st.session_state.out_png)).convert("RGBA")

        # output NON gigante: larghezza controllata
        st.image(out_img, width=520)

        st.caption("PNG trasparente 2000×2000")

        st.download_button(
            "Scarica PNG",
            data=st.session_state.out_png,
            file_name=st.session_state.out_filename or "output.png",
            mime="image/png",
        )

        # Info salvataggio
        st.caption(f"Salvato anche in locale: `{st.session_state.out_path}`")
        st.caption("Nota: su Streamlit Cloud il file resta nel server (temporaneo). Per il cliente conta il download.")

    else:
        st.info("Genera un output per vederlo qui.")
