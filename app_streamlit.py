import re
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

# OpenCV headless (serve per floodFill robusto)
import cv2


# -------------------------
# Utility
# -------------------------
def slugify(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def build_filename(brand: str, prodotto: str, anno: str) -> str:
    b = slugify(brand) or "brand"
    p = slugify(prodotto) or "prodotto"
    a = slugify(anno) or "anno"
    return f"{b}_{p}_{a}.png"


def ensure_unique_path(path: Path) -> Path:
    """Evita overwrite in locale: brand_prod_anno.png, brand_prod_anno_2.png, ..."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 2
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


# -------------------------
# Core: background removal robusto
# -------------------------
def remove_white_background_floodfill(
    rgb: np.ndarray,
    white_thr: int = 220,
    chroma_thr: int = 28,
) -> np.ndarray:
    """
    Rimuove SOLO il bianco collegato ai bordi (sfondo),
    così etichette chiare / riflessi interni non vengono tagliati.

    rgb: HxWx3 uint8
    return: alpha HxW uint8 (255 foreground, 0 background)
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    mn = np.minimum(np.minimum(r, g), b)
    mx = np.maximum(np.maximum(r, g), b)

    # Candidato sfondo: molto chiaro + poco "colorato" (quasi neutro)
    bg_candidate = (mn >= white_thr) & ((mx - mn) <= chroma_thr)
    bg_u8 = (bg_candidate.astype(np.uint8) * 255)

    h, w = bg_u8.shape

    # Flood fill su maschera: riempiamo SOLO le aree bg_candidate connesse ai bordi
    # cv2.floodFill richiede una mask più grande di 2 px
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    filled = bg_u8.copy()

    # Prova 4 angoli (se l’immagine ha margini strani)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for x, y in seeds:
        if filled[y, x] == 255:
            cv2.floodFill(filled, flood_mask, (x, y), 128)

    # Background reale = 128 (riempito) + eventuale bianco rimasto sui bordi
    bg_connected = (filled == 128)

    alpha = np.where(bg_connected, 0, 255).astype(np.uint8)
    return alpha


def feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return alpha
    a = Image.fromarray(alpha, mode="L")
    a = a.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.array(a, dtype=np.uint8)


def compose_2000_canvas(rgba: np.ndarray, target_size: int = 2000, bottle_ratio: float = 0.72) -> Image.Image:
    """
    Nessun autocrop: prende l'oggetto con trasparenza e lo ridimensiona
    su canvas 2000x2000.
    bottle_ratio = altezza oggetto rispetto al canvas (es. 0.72 = 72%)
    """
    img = Image.fromarray(rgba, mode="RGBA")

    # bounding box dell’alpha (solo per capire dimensioni oggetto)
    alpha = np.array(img.split()[-1])
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        # niente foreground -> ritorna canvas trasparente
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    # Crop SOLO per calcolare il ridimensionamento dell'oggetto (non è "autocrop finale")
    # serve a evitare di scalare anche tutta l'area vuota.
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    obj = img.crop((x0, y0, x1 + 1, y1 + 1))

    target_h = int(target_size * bottle_ratio)
    scale = target_h / max(1, obj.size[1])
    target_w = max(1, int(obj.size[0] * scale))

    obj_resized = obj.resize((target_w, target_h), resample=Image.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    x = (target_size - target_w) // 2
    y = (target_size - target_h) // 2
    canvas.alpha_composite(obj_resized, (x, y))
    return canvas


def process_image(
    pil_img: Image.Image,
    white_thr: int,
    feather: int,
    chroma_thr: int = 28,
) -> Image.Image:
    # converti in RGB
    rgb_img = pil_img.convert("RGB")
    rgb = np.array(rgb_img, dtype=np.uint8)

    # alpha robusto
    alpha = remove_white_background_floodfill(rgb, white_thr=white_thr, chroma_thr=chroma_thr)

    # feather bordi
    alpha = feather_alpha(alpha, feather)

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)

    # canvas finale 2000x2000, bottiglia ~72%
    out = compose_2000_canvas(rgba, target_size=2000, bottle_ratio=0.72)
    return out


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Vinitud Cutout Tool", layout="wide")

st.title("🍾 Vinitud Cutout Tool")
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (cloud / locale).")

# Stato
if "out_img" not in st.session_state:
    st.session_state.out_img = None
if "out_bytes" not in st.session_state:
    st.session_state.out_bytes = None
if "saved_path" not in st.session_state:
    st.session_state.saved_path = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

# Layout: sinistra stretta, destra output
left, right = st.columns([1, 2.2], gap="large")

with left:
    st.subheader("Input")

    uploaded = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    brand = st.text_input("Brand", value="")
    prodotto = st.text_input("Prodotto", value="")
    anno = st.text_input("Anno", value="")

    white_thr = st.slider("Soglia bianco", min_value=200, max_value=255, value=220, step=1)
    feather = st.slider("Morbidezza bordo", min_value=0, max_value=15, value=3, step=1)

    st.divider()
    st.caption("Preview input (piccolo)")

    if uploaded:
        pil_in = Image.open(uploaded)
        st.image(pil_in, caption="Input", use_container_width=True)
    else:
        pil_in = None
        st.info("Carica una foto per iniziare.")

    st.divider()

    # Bottone sempre visibile (stabile) con form
    with st.form("generate_form", clear_on_submit=False):
        submitted = st.form_submit_button("Genera output", use_container_width=True)

    if submitted:
        if pil_in is None:
            st.error("Carica prima un’immagine.")
        else:
            try:
                out = process_image(pil_in, white_thr=white_thr, feather=feather)

                # bytes per download
                buf = np.frombuffer(b"", dtype=np.uint8)  # placeholder
                import io
                bio = io.BytesIO()
                out.save(bio, format="PNG", optimize=True)
                out_bytes = bio.getvalue()

                filename = build_filename(brand, prodotto, anno)

                # salva anche in locale (su cloud è un disco temporaneo: non lo “vedi” nel tuo PC)
                out_dir = Path.home() / "VinitudCutout_Output"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = ensure_unique_path(out_dir / filename)
                out.save(out_path)

                st.session_state.out_img = out
                st.session_state.out_bytes = out_bytes
                st.session_state.saved_path = str(out_path)
                st.session_state.last_filename = out_path.name

                st.success("Output generato ✅")

            except Exception as e:
                st.session_state.out_img = None
                st.session_state.out_bytes = None
                st.session_state.saved_path = None
                st.session_state.last_filename = None
                st.error(f"Errore in elaborazione: {e}")

with right:
    st.subheader("Output")

    if st.session_state.out_img is None:
        st.info("Genera un output per vedere il PNG trasparente.")
    else:
        # Output non gigante: limitiamo la larghezza dentro la colonna
        st.image(st.session_state.out_img, caption="PNG trasparente 2000×2000", width=520)

        st.download_button(
            label="Scarica PNG",
            data=st.session_state.out_bytes,
            file_name=st.session_state.last_filename or "output.png",
            mime="image/png",
        )

        st.caption(
            f"Salvataggio server: `{st.session_state.saved_path}`\n\n"
            "Nota: su Streamlit Cloud questo percorso è nel server (non nel tuo computer). "
            "Per averlo sul PC usa **Scarica PNG**."
        )
