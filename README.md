# 🤖 Digit Recognition AI

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model Accuracy](https://img.shields.io/badge/Accuracy-99.1%25-brightgreen.svg)

## Overview
Digit Recognition AI is an interactive chat-style web app that recognizes handwritten digits (0–9) using a Convolutional Neural Network trained on the MNIST dataset.

The repository is organized for quick deployment and reproducibility. Place your pretrained model file at `model/digit_recognition_model.keras` for instant startup.

## Repository structure
```
digit-recognition-ai/
├── app.py
├── model/
│   └── digit_recognition_model.keras   # add your trained model here
├── notebook/
│   └── Train_Model_in_Colab.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick start (local)
1. Clone the repo:
   ```bash
   git clone https://github.com/abusufyan-netizen/digit-recognition-ai.git
   cd digit-recognition-ai
   ```
2. Create virtual environment and install:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\\Scripts\\activate # Windows
   pip install -r requirements.txt
   ```
3. Add your pretrained model:
   - Place `digit_recognition_model.keras` inside the `model/` folder.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Train in Google Colab (recommended)
1. Open `notebook/Train_Model_in_Colab.ipynb` in Google Colab.
2. Set Runtime → Change runtime type → GPU.
3. Run all cells. When training finishes the file `model/digit_recognition_model.keras` will be created.
4. Download the model from Colab and add it to the `model/` folder in this repository, then commit & push.

## Deploy to Streamlit Cloud
1. Push repository to GitHub.
2. Visit https://streamlit.io/cloud and create a new app from this repository.
3. Ensure `model/digit_recognition_model.keras` is present in the repo root inside `/model/` so the app loads instantly.

## Author
**Abu Sufyan — Student**  
Organization: Abu Zar  
GitHub: [@abusufyan-netizen](https://github.com/abusufyan-netizen)

## License
MIT License © 2025 Abu Sufyan
