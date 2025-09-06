import os
import time
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from io import BytesIO
from PIL import Image, ImageOps

import faiss
import torch

from src.embeddings import img_to_feature, crop_center_square
from src.search import search_similar_features
from src.utils import display_query_and_top_matches, get_avt_img, visualize_embeddings
from src.config import *

from facenet_pytorch import InceptionResnetV1


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Page configuration
st.set_page_config(page_title="Employee Dashboard", page_icon="🧑‍💼", layout="wide")

st.markdown("""
    <style>
        .employee-card {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 15px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .employee-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .status-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.8em;
            margin-top: 8px;
        }
        .checked-in {
            background-color: #d4edda;
            color: #155724;
        }
        .not-checked {
            background-color: #f8d7da;
            color: #721c24;
        }
        .top-match-card {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
            background: #f1f8f6;
            margin-bottom: 10px;
        }
        .other-match-card {
            border: 2px solid #FFC107;
            border-radius: 10px;
            padding: 10px;
            background: #fffbea;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🧑‍💼 Employee Check-in Dashboard</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_facenet_model():
    return InceptionResnetV1(pretrained="vggface2").eval()

# Load FAISS index + label map
@st.cache_resource
def load_faiss_index():
    if not os.path.exists(FEATURE_INDEX_PATH) or not os.path.exists(FEATURE_LABEL_MAP):
        st.error("❌ FAISS index or label map not found.")
        return None, None, None

    index = faiss.read_index(FEATURE_INDEX_PATH)
    label_map = np.load(FEATURE_LABEL_MAP)
    embeddings = index.reconstruct_n(0, index.ntotal)

    return index, label_map, embeddings

face_recognition_model = load_facenet_model()
index, label_map, all_embeddings = load_faiss_index()

# State
if "checkin_status" not in st.session_state:
    st.session_state.checkin_status = {name: False for name in label_map} if label_map is not None else {}

for key in ["captured_image", "matching_result", "matching_distance",
            "matching_avatar", "query_embedding", "all_matches", "capture_clicked"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "all_matches" else []


# Sidebar Navigation
page = st.sidebar.radio("📍 Navigate", ["👥 Employee List", "📸 Check-in", "📊 Analytics"])

# Employee List Page
if page == "👥 Employee List":
    st.markdown("### Employee Directory")

    if not st.session_state.checkin_status:
        st.warning("No employee data available.")
    else:
        cols = st.columns(3)
        for i, (name, checked) in enumerate(st.session_state.checkin_status.items()):
            with cols[i % 3]:
                avatar = get_avt_img(name)
                status_class = "checked-in" if checked else "not-checked"
                status_text = "✅ CHECKED IN" if checked else "❌ NOT CHECKED"

                image = Image.open(avatar).resize((250, 250))
                st.markdown(f"""
                    <div class="employee-card">
                        <img src="data:image/png;base64,{base64.b64encode(open(avatar, "rb").read()).decode()}" style="width:100%; border-radius:10px;" />
                        <h4>{name}</h4>
                        <div class="status-badge {status_class}">{status_text}</div>
                    </div>
                """, unsafe_allow_html=True)

# Check-in Page
elif page == "📸 Check-in":
    st.markdown("### 📸 Employee Check-in")

    col_cam, col_up = st.columns(2)
    with col_cam:
        if st.button("📷 Use Camera", use_container_width=True, type="primary"):
            st.session_state.capture_clicked = True
    with col_up:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.session_state.captured_image = crop_center_square(Image.open(uploaded_file).convert("RGB"))

    # Camera capture
    if st.session_state.capture_clicked:
        st.info("Opening camera... (3 second countdown)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access camera.")
        else:
            countdown_placeholder = st.empty()
            frame_placeholder = st.empty()
            start_time = time.time()
            capture_delay = 3
            captured_frame = None

            while (time.time() - start_time) < capture_delay:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame")
                    break
                remaining = int(capture_delay - (time.time() - start_time)) + 1
                countdown_placeholder.markdown(
                    f"<h2 style='text-align:center;color:#e91e63;font-size:48px;'>{remaining}</h2>",
                    unsafe_allow_html=True
                )
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                captured_frame = frame_rgb
                time.sleep(0.05)

            cap.release()
            countdown_placeholder.empty()
            frame_placeholder.empty()
            if captured_frame is not None:
                st.session_state.captured_image = crop_center_square(Image.fromarray(captured_frame))
                st.session_state.capture_clicked = False

    # Run recognition
    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Captured Image", use_container_width=True)

        with st.spinner("🔍 Identifying employee..."):
            query_embedding = img_to_feature(st.session_state.captured_image, face_recognition_model)
            st.session_state.query_embedding = query_embedding
            matches = search_similar_features(query_embedding, k=5, threshold=SIMILARITY_THRESHOLD)
            st.session_state.all_matches = matches

        if matches:
            best_match_name, best_distance, _ = matches[0]
            st.session_state.checkin_status[best_match_name] = True
            st.success(f"✅ {best_match_name} checked in (Similarity: {best_distance:.4f})")

            st.markdown("### 🏆 Top Matches")
            cols = st.columns(2)
            for i, (name, similarity, _) in enumerate(matches[:5]):
                card_class = "top-match-card" if i == 0 else "other-match-card"
                avatar = get_avatar_image(name)
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="{card_class}">
                            <img src="data:image/png;base64,{base64.b64encode(open(avatar, "rb").read()).decode()}" 
                                 style="width:100%; border-radius:8px;" />
                            <div style="margin-top:5px; text-align:center;">
                                <strong>{name}</strong><br>
                                <small>Similarity: {similarity:.4f}</small>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# ----------------------
# Analytics
# ----------------------
elif page == "📊 Analytics":
    st.markdown("### 📊 Embedding Space Analysis")
    st.markdown("Explore how employee embeddings are distributed in vector space.")

    if st.session_state.query_embedding is not None:
        visualize_embeddings(
            st.session_state.query_embedding,
            [(name, dist, idx) for name, dist, idx in st.session_state.all_matches[:5]]
        )
    else:
        visualize_embeddings()