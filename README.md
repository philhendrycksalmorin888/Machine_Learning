# Machine Learning–Based Detection of Malicious Network Traffic

Final project package for a Random Forest–based cybersecurity intrusion-detection prototype.

## Contents

- `notebooks/Machine_Learning_Cybersecurity_Final_Notebook.ipynb` — Jupyter/Google Colab notebook for full reproducibility.
- `streamlit_app.py` — Streamlit demonstration app.
- `src/cybersecurity_pipeline.py` — reusable training and evaluation pipeline.
- `assets/` — charts used in the final presentation.
- `requirements.txt` — Python dependencies.
- `GITHUB_STREAMLIT_LINKS.md` — placeholders and commands for publication links.

## Dataset

The project expects the course-supplied `cybersecurity.csv` file with the 13-column schema described in the documentation. The original public URL and license were not supplied, so the dataset should not be presented as NSL-KDD unless that source is verified.

## Run the Streamlit App Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Upload `cybersecurity.csv` inside the app to train the model and view evaluation results.

## Run the Notebook

Open `notebooks/Machine_Learning_Cybersecurity_Final_Notebook.ipynb` in Jupyter or Google Colab, then upload `cybersecurity.csv` when prompted.

## Main Reported Results

- Random Forest @ 0.50 threshold: 96.50% accuracy, 64.71% attack precision, 27.50% attack recall, 0.386 F1.
- Random Forest @ F2 threshold 0.229: 91.95% accuracy, 29.44% attack precision, 72.50% attack recall, 0.419 F1.
- PR-AUC / average precision: 0.534.

This is a coursework prototype and should not be used as a production intrusion-detection system without external and chronological validation.
