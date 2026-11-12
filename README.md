# Heart Disease Prediction — Streamlit

A Streamlit interface for a fitted scikit-learn heart-disease classification pipeline.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `heart_disease_complete_model.pkl` — add your **complete fitted pipeline** here

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and the model file.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy.

The model file must be a fitted scikit-learn `Pipeline` containing both `preprocessor` and `classifier` steps. The app validates this at startup and will stop with an explicit error if the uploaded file is preprocessing-only.

**Privacy and safety:** Do not enter identifiable patient information. This is an educational demonstration, not a medical device or diagnostic service.
