import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the trained model
model = keras.models.load_model("fish_sepcies.h5")

# Define a function to make predictions
def predict_species(image):
    img = np.array(image)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# Streamlit app
st.title("Fish Species Identification")

st.write("Upload an image to identify the fish species.")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Define species names for 9 classes
species_names = [
    "Black sea Sprat",
    "Gilt Head Bream",
    "Horse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Identify"):
        prediction = predict_species(image)
        predicted_class = np.argmax(prediction)
        
        species_name = species_names[predicted_class]
        
        st.write(f"Predicted Species: {species_name}")
        st.write(f"Prediction Accuracy: {prediction[0][predicted_class]:.2f}")
