import streamlit as st
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# Use a relative path so it works on Cloud and Mac
DRAWINGS_DIR = "drawings" 

st.set_page_config(page_title="Part ID Scanner", page_icon="⚙️")

def load_reference_drawings(directory):
    """Loads all valid images from the drawings directory."""
    drawings = {}
    if not os.path.exists(directory):
        return {}

    # Get all files, ignoring hidden system files (like .DS_Store)
    valid_files = [f for f in os.listdir(directory) if not f.startswith('.')]
    
    for filename in valid_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(directory, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                drawings[filename] = img
    return drawings

def identify_part(captured_image, reference_drawings):
    """Matches the captured photo against loaded drawings using ORB."""
    # Convert uploaded/captured file to OpenCV format
    file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
    img_query = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img_query is None:
        return None, 0

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img_query, None)
    
    if des1 is None:
        return None, 0

    best_match_name = None
    max_matches = 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for filename, img_train in reference_drawings.items():
        kp2, des2 = orb.detectAndCompute(img_train, None)
        
        if des2 is not None:
            matches = bf.match(des1, des2)
            # Strict filter: keep only "good" matches
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > max_matches:
                max_matches = len(good_matches)
                best_match_name = filename

    # Calculate rough confidence score
    confidence = min(100, (max_matches / 20) * 100)
    return best_match_name, confidence

# --- MAIN APP UI ---
st.title("⚙️ Shop Part Scanner")

# 1. Load Drawings
references = load_reference_drawings(DRAWINGS_DIR)

if not references:
    st.error(f"❌ No drawings found! Please make sure you have a folder named '{DRAWINGS_DIR}' in your GitHub repository and it contains images.")
    st.stop()
else:
    st.success(f"✅ System Ready: {len(references)} drawings loaded.")

# 2. Camera Input
img_file_buffer = st.camera_input("Take a picture of the part")

if img_file_buffer is not None:
    with st.spinner("Analyzing geometry..."):
        match_name, score = identify_part(img_file_buffer, references)
        
        if match_name and score > 15:
            st.markdown("---")
            st.header(f"Match: {match_name}")
            st.metric("Confidence", f"{int(score)}%")
            
            # Show Reference Image
            ref_path = os.path.join(DRAWINGS_DIR, match_name)
            st.image(ref_path, caption="Reference Drawing", width=300)
        else:
            st.warning("No clear match found. Try placing the part on a dark background.")