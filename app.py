import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Load the trained model
model = load_model("models/flower_model.h5")

# Class labels
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Preprocessing + prediction
def predict_flower(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)  # raw [0, 255]
    img_array = preprocess_input(img_array)  # normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    return {
        f"{class_names[i]} ({preds[i]:.4f})": float(preds[i])
        for i in range(len(class_names))
    }

# Gradio interface
demo = gr.Interface(
    fn=predict_flower,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🌸 Flower Classifier",
    description="Upload a flower image to classify it into one of the five categories: daisy, dandelion, roses, sunflowers, or tulips.",
)

if __name__ == "__main__":
    demo.launch()
