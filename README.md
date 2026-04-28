# NN Time Series Comparison

Comparison of **CNN**, **LSTM**, and **CNN-LSTM hybrid** models for multivariate time-series forecasting on the **Jena Climate dataset** using **PyTorch**.

## Project Goal

The goal of this project is to compare three neural network architectures for temperature prediction from multivariate weather time series:

- **1D-CNN**
- **LSTM**
- **CNN-LSTM hybrid**

We focus on:
- prediction accuracy (**RMSE**, **MAE**),
- effect of input window size (**24 vs 48**),
- number of parameters,
- training time,
- convergence stability,
- and qualitative error analysis.

## Team

- **Person 1**: Data pipeline, shared training infrastructure, LSTM
- **Person 2**: CNN
- **Person 3**: CNN-LSTM Hybrid
- **Person 4**: Evaluation, plots, analysis, report integration

## Dataset

We use the **Jena Climate Dataset**.

Dataset source:
```text
https://www.kaggle.com/datasets/mnassrib/jena-climate
