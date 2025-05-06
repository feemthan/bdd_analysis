import math
import os
import re

import streamlit as st
import yaml
from PIL import Image

# ——— CONFIGURATION ———
IMAGE_DIR = os.path.join("output_files", "bdd_yolo_test")
CONFIG_PATH = os.path.join("artifacts", "args.yaml")

st.set_page_config(page_title="YOLO Detection Results", layout="wide")
st.title("YOLO Detection Results Viewer")


@st.cache_data(ttl=60)
def list_images():
    """Return a sorted list of image filenames in IMAGE_DIR."""
    if not os.path.isdir(IMAGE_DIR):
        return []
    files = [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    def key_fn(fn):
        m = re.match(r"results_(\d+)\.", fn)
        return int(m.group(1)) if m else fn.lower()

    return sorted(files, key=key_fn)


@st.cache_data(ttl=60)
def load_image(fn):
    """Open an image from disk."""
    return Image.open(os.path.join(IMAGE_DIR, fn))


@st.cache_data(ttl=300)
def load_config():
    """Load YOLO configuration from artifacts/args.yaml."""
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ——— INITIALIZE STATE ———
images = list_images()
if not images:
    st.warning(f"No images found in `{IMAGE_DIR}`.")
    st.stop()


def _init_state(key, val):
    if key not in st.session_state:
        st.session_state[key] = val


_init_state("view_mode", "Single Image")
_init_state("selected", images[0])
_init_state("per_page", 8)
_init_state("page", 1)


# ——— CALLBACKS ———
def select_image(fn):
    st.session_state.selected = fn


def view_image(fn):
    st.session_state.selected = fn
    st.session_state.view_mode = "Single Image"


# ——— SIDEBAR ———
with st.sidebar:
    st.info(f"Found **{len(images)}** images")
    st.radio("View Mode", ["Single Image", "Gallery View"], key="view_mode")
    if st.session_state.view_mode == "Gallery View":
        st.slider("Images per page", min_value=4, max_value=20, step=1, key="per_page")
        pages = math.ceil(len(images) / st.session_state.per_page)
        st.number_input("Page", min_value=1, max_value=pages, key="page")
        st.markdown(f"Page **{st.session_state.page}** of **{pages}**")

# ——— SINGLE IMAGE VIEW ———
if st.session_state.view_mode == "Single Image":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Select Image")
        term = st.text_input("Search filename")
        opts = [fn for fn in images if term.lower() in fn.lower()] if term else images
        st.selectbox("Choose:", opts, key="selected")

    with col2:
        fn = st.session_state.selected
        st.subheader(fn)
        st.image(load_image(fn), use_container_width=True)

        m = re.match(r"results_(\d+)\.", fn)
        if m:
            idx = int(m.group(1))
            prev_fn = f"results_{idx-1}.jpg"
            next_fn = f"results_{idx+1}.jpg"
            c1, c2 = st.columns(2)
            with c1:
                if prev_fn in images:
                    st.button(
                        "← Previous",
                        on_click=select_image,
                        args=(prev_fn,),
                        key=f"prev_{prev_fn}",
                    )
            with c2:
                if next_fn in images:
                    st.button(
                        "Next →",
                        on_click=select_image,
                        args=(next_fn,),
                        key=f"next_{next_fn}",
                    )

# ——— GALLERY VIEW ———
else:
    st.subheader("Gallery View")
    ipp = st.session_state.per_page
    start = (st.session_state.page - 1) * ipp
    page_imgs = images[start : start + ipp]
    rows = math.ceil(len(page_imgs) / 2)
    for r in range(rows):
        cols = st.columns(2)
        for c, col in enumerate(cols):
            idx = r * 2 + c
            if idx < len(page_imgs):
                fn = page_imgs[idx]
                with col:
                    st.image(load_image(fn), caption=fn, use_container_width=True)
                    st.button(
                        "View →", on_click=view_image, args=(fn,), key=f"view_{fn}"
                    )

# ——— YOLO CONFIGURATION ———
with st.expander("YOLO Configuration", expanded=False):
    cfg = load_config()
    if cfg:
        st.code(yaml.dump(cfg, default_flow_style=False), language="yaml")
    else:
        st.warning(f"Could not load YOLO configuration from `{CONFIG_PATH}`.")
