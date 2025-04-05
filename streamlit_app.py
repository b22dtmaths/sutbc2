import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="BreakHis Classification", page_icon="🔬")

# Class names ตามลำดับของโฟลเดอร์ใน training data
CLASS_NAMES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('breakhis_model_wavelet.h5')

def preprocess_image(image):
    # Resize
    image = image.resize((224, 224))
    # Convert to array and add batch dimension
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    # Normalize
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = tf.expand_dims(image_array, 0)
    return image_array

def predict_image(image_array, model):
    predictions = model.predict(image_array)
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1] #top_3_idx = np.argsort(predictions[0])[-8:][::-1]
    top_3_values = predictions[0][top_3_idx]
    results = []
    for idx, value in zip(top_3_idx, top_3_values):
        results.append({
            'class': CLASS_NAMES[idx],
            'probability': float(value)
            
        })
    #st.write(predictions)
    return results
    
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🔬 BreakHis Image Classification")
#st.write("Upload a histopathology image for classification")
st.write('Upload a histopathology image for classification '
         '[Click to view a sample image](https://bit.ly/sambhimg)')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            # Preprocess
            processed_image = preprocess_image(image)
            # Get predictions
            results = predict_image(processed_image, model)
            
            # Display results
            st.success("Analysis Complete!")
            st.subheader("Top 3 Predictions:")
            
            # Create columns for each prediction
            cols = st.columns(3) #cols = st.columns(4)
            for idx, result in enumerate(results):
                with cols[idx]: #with cols[idx%4]:
                    st.metric(
                        label=result['class'].replace('_', ' ').title(),
                        value=f"{result['probability']:.1%}"
                    )
                    
            # Display confidence bars
            st.subheader("Confidence Levels:")
            for result in results:
                st.write(f"{result['class'].replace('_', ' ').title()}")
                st.progress(float(result['probability']))
