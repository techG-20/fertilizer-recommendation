# 🌱 Fertilizer Recommendation System

An end-to-end machine-learning + agronomic-rule fertilizer recommendation project.

## Features
- N, P, K + temperature + humidity + soil moisture inputs
- Soil and crop selection
- Nutrient-gap analysis
- Explainable fertilizer recommendation
- Top-3 rule-based candidates
- Optional Random Forest prediction
- Streamlit dashboard
- Unit tests
- Training script

> This is decision-support software, not a substitute for laboratory soil testing or local agronomic advice.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Train the demo ML model:

```bash
python -m src.train
```

## Dataset

The included `data/demo_fertilizer.csv` is **synthetic and only for software demonstration**.

For a real ML experiment, replace it with a properly licensed fertilizer dataset. A relevant public dataset is the Kaggle Fertilizer Prediction dataset:

https://www.kaggle.com/datasets/irakozekelly/fertilizer-prediction

It uses fields such as temperature, humidity, soil moisture, soil type, crop type, nitrogen, phosphorus, potassium and fertilizer name.

A public Hugging Face dataset with the same general schema is also available:

https://huggingface.co/datasets/kaifahmad/Fertilizer-Prediction

Expected columns:

```text
Temperature,Humidity,Moisture,Soil Type,Crop Type,
Nitrogen,Phosphorous,Potassium,Fertilizer Name
```

## Architecture

```text
Streamlit UI
     |
Input validation
     |
     +----> Nutrient-gap engine ----> explainable recommendation
     |
     +----> Random Forest ----------> top-3 ML predictions
```

## Fertilizer knowledge base

| Fertilizer | N-P-K |
|---|---:|
| Urea | 46-0-0 |
| DAP | 18-46-0 |
| 14-35-14 | 14-35-14 |
| 28-28 | 28-28-0 |
| 17-17-17 | 17-17-17 |
| 20-20 | 20-20-0 |
| 10-26-26 | 10-26-26 |

These labels are product-style compositions represented in the reference datasets; actual commercial formulations should be checked on the product label.

## Portfolio upgrades
- SHAP explanations
- XGBoost model comparison
- weather API
- IoT soil sensors
- crop-stage-aware recommendations
- fertilizer price comparison
- multilingual UI
- Docker
- GitHub Actions CI/CD
- model versioning

## License
MIT for the original project code. Dataset licenses remain with their respective providers.
