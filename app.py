import streamlit as st
import cv2
import numpy as np
import os

# ================== Feature Extraction ==================
def extract_features(image):
    # Convert to HSV and compute color histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture feature using Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Shape feature - sum of contour areas
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_feature = sum(cv2.contourArea(c) for c in contours)

    # Combine features
    return np.hstack([hist, texture, shape_feature])

# ================== Similarity Calculation ==================
def calculate_similarity(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)

# ================== Streamlit App ==================
st.set_page_config(page_title="Real-Time CBIR", layout="centered")
st.title("📸 Real-Time Content-Based Image Retrieval (CBIR)")
st.write("Store images in a live database and retrieve similar ones without using datasets or pre-trained models.")

# Ensure storage folder exists
os.makedirs("saved_frames", exist_ok=True)

# Create in-memory database
if "db" not in st.session_state:
    st.session_state.db = {}  # {filename: feature_vector}

# ----------- Image Capture from Camera -----------
st.subheader("📥 Add Images to Database")
camera = st.camera_input("Capture an Image")

if camera:
    file_path = f"saved_frames/frame_{len(st.session_state.db)+1}.jpg"
    with open(file_path, "wb") as f:
        f.write(camera.getbuffer())

    image = cv2.imread(file_path)
    features = extract_features(image)

    st.session_state.db[file_path] = features
    st.success(f"✅ Image added! Database size: {len(st.session_state.db)}")

# ----------- Query Section -----------
st.subheader("🔍 Search for Similar Images")
query_img = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])

if query_img:
    file_bytes = np.asarray(bytearray(query_img.read()), dtype=np.uint8)
    query_image = cv2.imdecode(file_bytes, 1)

    # Extract query features
    query_features = extract_features(query_image)

    # Compare with database
    results = []
    for fname, feats in st.session_state.db.items():
        sim = calculate_similarity(query_features, feats)
        results.append((fname, sim))

    # Sort by similarity
    results.sort(key=lambda x: x[1])

    # Show results
    st.subheader("📂 Top Matches")
    if results:
        for fname, sim in results[:5]:
            st.image(fname, caption=f"Similarity Score: {sim:.2f}")
    else:
        st.warning("⚠ No images in the database yet. Please add some first.")

# ----------- Footer -----------
st.markdown("---")
st.caption("Developed for FI1917 - Content Based Information Retrieval | Using OpenCV + Streamlit")
