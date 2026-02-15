import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
# Use a relative path so it works on Cloud and Mac
DRAWINGS_DIR = "drawings" 

st.set_page_config(page_title="Part ID Scanner", page_icon="⚙️")

def load_reference_drawings(directory):
    """Loads all valid images from the drawings directory."""
    drawings = {}
    
    # Check if directory exists (Create it if not, to avoid errors)
    if not os.path.exists(directory):
        st.error(f"⚠️ Folder '{directory}' not found. Please create it in your GitHub repo.")
        return {}

    # Get all files, ignoring hidden system files
    valid_files = [f for f in os.listdir(directory) if not f.startswith('.')]
    
    for filename in valid_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(directory, filename)
            # Read image in Grayscale (0)
            img = cv2.imread(path, 0)
            if img is not None:
                drawings[filename] = img
                
    return drawings

def identify_part(file_buffer, reference_drawings):
    """Matches the uploaded/captured photo against loaded drawings."""
    # Convert the file buffer (from camera or upload) into an OpenCV image
    file_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
    img_query = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img_query is None:
        return None, 0

    # Initialize ORB detector (The "Eye")
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img_query, None)
    
    if des1 is None:
        return None, 0

    best_match_name = None
    max_matches = 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Compare against every drawing in the folder
    for filename, img_train in reference_drawings.items():
        kp2, des2 = orb.detectAndCompute(img_train, None)
        
        if des2 is not None:
            matches = bf.match(des1, des2)
            # Strict filter: keep only "good" matches (closer distance is better)
            good_matches = [m for m in matches if m.distance < 60]
            
            if len(good_matches) > max_matches:
                max_matches = len(good_matches)
                best_match_name = filename

    # Calculate confidence score based on number of feature matches
    # (Adjust '25' if you need it to be more/less strict)
    confidence = min(100, (max_matches / 25) * 100)
    return best_match_name, confidence

# --- MAIN APP UI ---
st.title("⚙️ Shop Part Scanner")

# 1. Load Drawings
with st.spinner("Loading shop drawings..."):
    references = load_reference_drawings(DRAWINGS_DIR)

if not references:
    st.warning("No drawings loaded yet. Upload images to your 'drawings' folder in GitHub!")
else:
    st.caption(f"✅ System Ready: {len(references)} parts indexed.")

# 2. Input Method (Tabs for cleaner look)
tab1, tab2 = st.tabs(["📷 Camera", "📤 Upload File"])

image_source = None

with tab1:
    camera_img = st.camera_input("Take a picture")
    if camera_img:
        image_source = camera_img

with tab2:
    uploaded_img = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_img:
        image_source = uploaded_img

# 3. Process the Image
if image_source is not None:
    st.divider()
    with st.spinner("Analyzing geometry..."):
        match_name, score = identify_part(image_source, references)
        
        if match_name and score > 15:
            st.success(f"### Match Found: {match_name}")
            st.metric("Confidence Score", f"{int(score)}%")
            
            # Show the Reference Drawing from the folder
            ref_path = os.path.join(DRAWINGS_DIR, match_name)
            
            # Display images side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_source, caption="Your Scan", width=200)
            with col2:
                st.image(ref_path, caption="Shop Drawing", width=200)
        else:
            st.error("No clear match found.")
            st.info("Try placing the part on a dark background or reducing glare.")
