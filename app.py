import streamlit as st
import numpy as np
from PIL import Image
import tensorflow 
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50

# Function to download the model weights file
def download_model_weights(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Download the model weights if not already present
model_weights_url = "https://github.com/omagarwal2002/Celebal-Technologies-Internship/raw/main/resnet_model.h5"
local_model_weights_path = "resnet_model.h5"
download_model_weights(model_weights_url, local_model_weights_path)

# Function to read and preprocess the image
def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160, 160)))


#our model
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(15, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the pre-trained ResNet model
resnet_model = model
resnet_model.load_weights(local_model_weights_path)

# Action label mapping
label_map = {
    0: "sitting",
    1: "using laptop",
    2: "hugging",
    3: "sleeping",
    4: "drinking",
    5: "clapping",
    6: "dancing",
    7: "cycling",
    8: "calling",
    9: "laughing",
    10: "eating",
    11: "fighting",
    12: "listening_to_music",
    13: "running",
    14: "texting"
}

# Function to make prediction
def make_prediction(test_image):
    result = resnet_model.predict(np.asarray([read_image(test_image)]))
    predicted_class = np.argmax(result)
    probability = np.max(result) * 100
    return predicted_class, probability



# Streamlit app
def main():
    st.title("Human Action Recognition")
    st.write("Upload an image and let the model predict the action present in it.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        predicted_class, probability = make_prediction(uploaded_image)


        # Display the prediction
        st.write(f"Predicted Action: {label_map[predicted_class]}")
        st.write(f"Probability: {probability:.2f}%")

if __name__ == "__main__":
    main()
