import io
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Vinitud Cutout Tool",
    page_icon="🍾",
    layout="centered",
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# UTILS
# =========================
def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "vinitud"


def pil_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def ensure_rgba(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGBA":
        return pil_img.convert("RGBA")
    return pil_img


def process_cutout(
    uploaded_file,
    white_threshold: int = 220,
    edge_softness: int = 3,
    out_size: int = 2000,
) -> Image.Image:
    """
    Assunzione: input con sfondo bianco (o quasi).
    Crea maschera alpha basata su soglia del bianco + morbidezza bordi.
    Output: PNG RGBA 2000x2000 con trasparenza.
    """
    # Leggi bytes -> OpenCV BGR
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Immagine non valida o non leggibile.")

    # Converti in RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Calcola "quanto è bianco" un pixel (max su canali o min su canali)
    # Usiamo min(R,G,B): se min è alto, pixel è vicino al bianco
    min_rgb = np.min(rgb, axis=2)

    # Maschera: dove NON è bianco -> oggetto
    # Oggetto = min_rgb < white_threshold
    mask = (min_rgb < white_threshold).astype(np.uint8) * 255

    # Pulizia maschera (morph)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Morbidezza bordo
    if edge_softness and edge_softness > 0:
        k = int(edge_softness) * 2 + 1  # kernel dispari
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # Crea RGBA: RGB + alpha(mask)
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask

    # Converti a PIL
    out = Image.fromarray(rgba, mode="RGBA")

    # Resize su canvas 2000x2000 mantenendo proporzioni + centratura
    # Prima: scala per entrare in out_size (con un po' di margine)
    target = out_size
    margin = int(target * 0.08)  # 8% margine
    max_w = target - margin * 2
    max_h = target - margin * 2

    w, h = out.size
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    out_resized = out.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas.paste(out_resized, (x, y), out_resized)

    return canvas


# =========================
# UI
# =========================
st.title("Vinitud Cutout Tool")
st.caption("Sfondo bianco → PNG trasparente 2000×2000, bottiglia ~72% (cloud).")

col_out, col_in = st.columns([3, 1.6], gap="large")

with col_in:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Carica immagine (sfondo bianco)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    brand = st.text_input("Brand", value="")
    prodotto = st.text_input("Prodotto", value="")
    anno = st.text_input("Anno", value="")

    white_threshold = st.slider(
        "Soglia bianco",
        min_value=200,
        max_value=255,
        value=220,
        help="Più basso = togli più sfondo (attenzione a perdere dettagli chiari).",
    )

    edge_softness = st.slider(
        "Morbidezza bordo",
        min_value=0,
        max_value=15,
        value=3,
        help="Aumenta per bordi più soft. 0 = bordi netti.",
    )

    genera = st.button("Genera output", type="primary", use_container_width=True)

    if uploaded_file:
        st.divider()
        st.caption("Preview input")
        st.image(uploaded_file, width=260)

with col_out:
    st.subheader("Output")

    if not uploaded_file:
        st.info("Carica una foto per iniziare.")
    else:
        if genera:
            try:
                out_img = process_cutout(
                    uploaded_file=uploaded_file,
                    white_threshold=white_threshold,
                    edge_softness=edge_softness,
                    out_size=2000,
                )

                # Nome file “pulito”
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{slugify(brand)}_{slugify(prodotto)}_{slugify(anno)}_{ts}.png"
                fname = fname.replace("__", "_").strip("_")
                if fname.startswith("_"):
                    fname = fname[1:]

                # Salva su disco (utile in locale). In Streamlit Cloud resta nel container.
                out_path = OUTPUT_DIR / fname
                out_img.save(out_path, format="PNG")

                # Mostra output non gigante
                st.image(out_img, caption="PNG trasparente 2000×2000", width=420)

                # Download (questo è quello che conta per il cliente)
                st.download_button(
                    "Scarica PNG",
                    data=pil_to_bytes(out_img),
                    file_name=fname,
                    mime="image/png",
                    use_container_width=False,
                )

                st.caption(f"File generato: `{fname}`")

            except Exception as e:
                st.error(f"Errore durante la generazione: {e}")

        else:
            st.warning("Regola i parametri e premi **Genera output**.")
