# 🌸 Flower Classification

A deep learning web app that identifies flowers from photos across 5 categories: **daisy, dandelion, rose, sunflower, and tulip**.

**🔗 Live demo:** _add your Streamlit app URL here once deployed_

## How it works

Upload a photo of a flower, and the model predicts which of the 5 categories it belongs to, along with a confidence score.

## Model

- **Architecture:** ResNet50 (transfer learning), pretrained on ImageNet
- **Training strategy:** Two-stage — frozen base training, followed by fine-tuning the top 20 layers
- **Input:** 224×224 RGB images
- **Dataset:** [TensorFlow flower_photos dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) (daisy, dandelion, roses, sunflowers, tulips), 200 images/class
- **Validation accuracy:** ~89.5%

Training and experimentation are documented in [`AIML2.ipynb`](./AIML2.ipynb).

## Tech stack

- **TensorFlow / Keras** — model training and inference
- **Streamlit** — web app interface
- **Hugging Face Hub** — model hosting (the trained model file is too large for GitHub's 100MB limit, so it's hosted separately and downloaded at runtime)

## Running locally

```bash
git clone https://github.com/medhardatta/Flower_classification.git
cd Flower_classification
pip install -r requirements.txt
streamlit run app.py
```

The app will download the trained model from Hugging Face Hub on first run and cache it locally.

## Project structure

```
├── app.py               # Streamlit web app
├── requirements.txt     # Python dependencies
├── AIML2.ipynb          # Model training notebook
└── README.md
```

## Confusion matrix

The model performs best on sunflowers (95.3% recall) and dandelions (94.7% recall). The most common confusion is between daisy and dandelion, which is expected given their visual similarity.
