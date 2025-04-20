# Phishing Email and Fake QR Code Detector

This project detects phishing emails and fake QR codes by analyzing email content, embedded URLs, and QR code data.

## Project Structure
- `main.py` — Streamlit frontend application.
- `data_preprocessing.py` — Preprocessing functions for email and URL data.
- `model_training.py` — Training script for email phishing detection model.
- `train_url_model.py` — Training script for URL phishing detection model.
- `email_model_accuracy.txt` and `url_model_accuracy.txt` — Model performance reports.

## Datasets Used
- [Phishing Emails Dataset (Kaggle)](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- [Phishing Emails Dataset (Zenodo)](https://zenodo.org/records/13474746)
- [Phishing URLs Dataset (Kaggle)](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls?resource=download)

## Note:
- **Model files (.pkl)** and **dataset files (.csv)** have been excluded from this repository due to GitHub's file size limitations.
- If needed, you can retrain the models using the provided scripts (`model_training.py` and `train_url_model.py`).
- To obtain the original datasets, please refer to the links provided above.

---

⭐
