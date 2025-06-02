---
title: 🌼 EfficientNetV2B0 Flower Classifier
emoji: 🌸
colorFrom: yellow
colorTo: pink
sdk: gradio
app_file: app.py
pinned: true
---
[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?logo=huggingface&style=flat-square)](https://huggingface.co/spaces/McKlay/efficientnet-flower-classifier)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange?logo=gradio&style=flat-square)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GitHub last commit](https://img.shields.io/github/last-commit/McKlay/TensorFlow-Companion-Book)
![GitHub Repo stars](https://img.shields.io/github/stars/McKlay/TensorFlow-Companion-Book?style=social)
![GitHub forks](https://img.shields.io/github/forks/McKlay/TensorFlow-Companion-Book?style=social)
![MIT License](https://img.shields.io/github/license/McKlay/TensorFlow-Companion-Book)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=McKlay.TensorFlow-Companion-Book)

# 🌼 EfficientNetV2B0 Flower Classifier

An elegant and efficient image classifier trained to recognize 5 flower types: **daisy**, **dandelion**, **roses**, **sunflowers**, and **tulips**.  
Powered by **TensorFlow**, fine-tuned using **EfficientNetV2B0**, and deployed with **Gradio** on **Hugging Face Spaces**.

![Model Accuracy](https://img.shields.io/badge/Validation_Accuracy-91%25-brightgreen)
![Made with TensorFlow](https://img.shields.io/badge/Built_with-TensorFlow-ff6f00?logo=tensorflow)
![Gradio UI](https://img.shields.io/badge/Interface-Gradio-20c997?logo=gradio)

---

## Live Demo

👉 **Try the app here**: [Hugging Face Space](https://huggingface.co/spaces/McKlay/efficientnet-flower-classifier)

Upload a flower image and get the top 5 predictions with confidence scores.

---

## Model Details

- **Backbone**: EfficientNetV2B0 (`keras.applications`)  
- **Framework**: TensorFlow 2.x  
- **Dataset**: TensorFlow Flowers (~3,700 images, 5 classes)  
- **Classes**: `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`  
- **Validation Accuracy**: **91.28%**  
- **Training Strategy**:  
  - Stage 1: 5 epochs (base frozen)  
  - Stage 2: 5 epochs (fine-tuning all layers)  
- **Preprocessing**: `preprocess_input()` scaled to [-1, 1]

---

## 📓 Training Notebooks

✅ Kaggle: [Flower Recognition – Fine-Tuning EfficientNetV2B0](https://www.kaggle.com/code/claymarksarte/flower-recognition-fine-tuning)  
Full training notebook with dataset loading, preprocessing, model building, and evaluation.

⚠️ Colab: (Archived) Training started in [Google Colab](https://colab.research.google.com/) but was moved to Kaggle due to GPU quota limitations.  
You can still view the original Colab notebook here: [Colab Fine-Tuning](https://colab.research.google.com/drive/1fSrxw2Pi48Adu25s1BcQFr2MnkLOCNzH?usp=sharing) 

--

## ## 📁 Project Structure

efficientnet-flower-classifier/  
├── app.py # Gradio app (entry point)  
├── models/  
│ └── flower_model.h5 # Trained Keras model  
├── requirements.txt  
└── README.md  

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/8_FlowerRecognition-HF.git  
cd 8_FlowerRecognition-HF  
pip install -r requirements.txt  
python app.py
```

---

## Dependencies

- tensorflow  
- gradio  
- numpy  
- pillow

---

## Acknowledgments

- [TensorFlow Flower Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)  

- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298) — Tan & Le

---

## 🧑‍💻 Author

Clay Mark Sarte  
[GitHub](https://github.com/McKlay) | [LinkedIn](https://www.linkedin.com/in/clay-mark-sarte-283855147/)

