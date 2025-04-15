import streamlit as st
from colorthief import ColorThief
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import io

# ========== Sample training data ========== 
# Format: (shirt RGB, pant RGB, match label)
sample_data = [
    # red shirt combinations
    ((255, 0, 0), (0, 0, 0), 1),      # red shirt - black pants
    ((255, 0, 0), (0, 128, 0), 0),    # red shirt - green pants (no match)
    ((255, 0, 0), (128, 128, 128), 1),  # red shirt - grey pants
    ((255, 0, 0), (255, 255, 255), 1),  # red shirt - white pants
    ((255, 0, 0), (165, 42, 42), 1),  # red shirt - brown pants

    # Add more shirt-pants combinations as needed...
]

# ========== Model training ========== 
# Prepare the training data
X = [list(shirt) for shirt, pant, label in sample_data]  # Only shirt colors for input
y = [pant for shirt, pant, label in sample_data]  # Corresponding pants colors as output

# Use KNeighborsClassifier to predict pant colors based on shirt color
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# ========== Functions ==========

def extract_dominant_color(img_file):
    img = Image.open(img_file)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = io.BytesIO(img_byte_arr.getvalue())
    color_thief = ColorThief(img_byte_arr)
    dominant_color = color_thief.get_color(quality=1)  # Get the dominant color
    return dominant_color

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def find_closest_shirt_color(shirt_rgb):
    """Find the closest shirt color in the dataset using Euclidean distance"""
    closest_shirt = None
    min_distance = float('inf')
    
    for shirt, pant, label in sample_data:
        distance = np.linalg.norm(np.array(shirt_rgb) - np.array(shirt))  # Euclidean distance between colors
        if distance < min_distance:
            min_distance = distance
            closest_shirt = shirt
    
    return closest_shirt

def recommend_pants_based_on_shirt(shirt_rgb):
    """Find pants based on the closest matching shirt color from the dataset"""
    closest_shirt = find_closest_shirt_color(shirt_rgb)
    recommendations = []
    
    for shirt, pant, label in sample_data:
        if shirt == closest_shirt and label == 1:  # If the shirt matches and is a valid match
            recommendations.append(pant)
    
    return recommendations

# ========== Streamlit UI ========== 
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")
st.title("👗 AI Fashion Stylist")
st.write("Upload a shirt image and get pant color suggestions!")

uploaded_file = st.file_uploader("Upload a shirt image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Shirt", use_container_width=True)
    
    # Extract the dominant color from the uploaded image
    shirt_rgb = extract_dominant_color(uploaded_file)
    st.write(f"🎨 Detected shirt color (RGB): {shirt_rgb} | HEX: {rgb_to_hex(shirt_rgb)}")
    
    # Recommend pants based on the extracted shirt color
    suggested_pants = recommend_pants_based_on_shirt(shirt_rgb)
    if suggested_pants:
        st.subheader("👖 Recommended Pant Colors:")
        for pant in suggested_pants:
            st.color_picker(f"Pants Color (RGB: {pant})", value=rgb_to_hex(pant), key=str(pant))
    else:
        st.warning("No matching pants found in dataset.")
