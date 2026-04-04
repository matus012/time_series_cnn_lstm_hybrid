# NN Time Series Comparison

Comparing CNN, LSTM, and CNN-LSTM hybrid on Jena Climate dataset (temperature prediction). PyTorch.

## Team
- Person 1: Data pipeline, shared training infra, LSTM
- Person 2: CNN
- Person 3: CNN-LSTM Hybrid
- Person 4: Evaluation & analysis

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Download Jena Climate dataset CSV into data/jena_climate_2009_2016.csv

## Run
python src/train.py
