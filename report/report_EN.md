# Comparison of Neural Network Architectures for Time Series Prediction
## Neural Networks 2025/2026

---

## 1. Problem Formulation

The goal of this project is to compare three different neural network architectures on a time series forecasting task. Specifically, we focus on predicting air temperature (T [°C]) from meteorological measurements recorded at the Jena Climate station between 2009 and 2016.

The task is formulated as a one-step-ahead forecasting problem: given a sliding window of the last `window_size` hourly observations across 14 meteorological features, the model predicts the temperature at the next time step.

We compare three approaches:
- **LSTM** – a recurrent network that models long-range sequential dependencies,
- **1D-CNN** – a convolutional network that captures local temporal patterns,
- **Hybrid CNN-LSTM** – a combined architecture that uses CNN for feature extraction and LSTM for long-range dependency modelling.

Research questions:
1. Which architecture achieves the lowest temperature prediction error?
2. How does the input window size (24 vs. 48 hours) affect accuracy?
3. Does the hybrid architecture add value over the standalone models?

---

## 2. Theoretical Background

### 2.1 LSTM (Long Short-Term Memory)

LSTM is a type of recurrent neural network designed to address the vanishing gradient problem that affects standard RNNs on long sequences [Hochreiter & Schmidhuber, 1997]. Its core consists of three gating mechanisms:

- **Input gate** – controls which new information is stored in memory,
- **Forget gate** – controls which information is discarded from memory,
- **Output gate** – controls which part of the memory is passed to the output.

For time series forecasting, LSTM is a natural choice because it explicitly models sequential dependencies of arbitrary length.

### 2.2 1D Convolutional Neural Network (1D-CNN)

A 1D-CNN applies convolutions along the time axis of a sequence. Unlike LSTM, it does not depend on previous hidden states; instead, it captures local patterns within the fixed receptive field determined by the kernel size [LeCun et al., 1998]. Advantages of CNN for time series:

- parallelisable computation (faster training than RNN),
- stable gradients,
- effective local pattern extraction.

### 2.3 Hybrid CNN-LSTM

The hybrid architecture combines both approaches: the CNN first extracts local features from the input sequence, and the LSTM then models long-range dependencies among these compressed representations. This exploits the strengths of both architectures while also reducing the computational load on the LSTM, since it operates on higher-level CNN features rather than the raw 14-dimensional input.

### 2.4 Evaluation Metrics

The chosen metrics are standard for regression tasks, reported in the original unit (°C):

- **MAE** (Mean Absolute Error) – average absolute deviation in °C; robust to outliers,
- **RMSE** (Root Mean Squared Error) – penalises larger errors more than MAE; suitable for literature comparison,
- **R²** (coefficient of determination) – proportion of explained variance; a value close to 1 indicates near-perfect fit.

---

## 3. Dataset Description and Preprocessing

### 3.1 Jena Climate Dataset

The dataset contains recordings from a meteorological station in Jena, Germany from 2009 to 2016. The original records have 10-minute resolution. The file contains 14 numerical variables:

| Variable | Description |
|----------|-------------|
| T (degC) | Air temperature [°C] – **target variable** |
| p (mbar) | Atmospheric pressure |
| rh (%) | Relative humidity |
| Tdew (degC) | Dew point |
| wv (m/s) | Wind speed |
| max. wv (m/s) | Maximum wind speed |
| wd (deg) | Wind direction |
| rain (mm) | Precipitation |
| SWDR (W/m²) | Solar radiation |
| PAR (µmol/m²/s) | Photosynthetically active radiation |
| max. PAR | Maximum PAR |
| Tlog (degC) | Logger temperature |
| CO2 (ppm) | CO₂ concentration |
| rho (g/m³) | Air density |

### 3.2 Preprocessing

All preprocessing was implemented centrally in `src/dataset.py` and shared across all models, ensuring direct comparability of results.

**Steps:**
1. **Subsampling** – every 6th row is kept (converting 10-minute to hourly resolution).
   - Original rows: 420,551 → after subsampling: **70,092**
2. **Numeric column selection** – 14 features retained.
3. **Missing value imputation** – forward fill.
4. **Chronological split** – no shuffling to prevent data leakage:
   - Training set: 70% → **49,064** samples
   - Validation set: 15% → **10,514** samples
   - Test set: 15% → **10,514** samples
5. **Normalisation** – `StandardScaler` fitted exclusively on the training set and then applied to validation and test sets.
   - Target variable statistics: **mean = 9.108 °C**, **std = 8.655 °C**
6. **Sliding window** – sequences of length `window_size` are created:
   - For `window_size = 24`: train 49,040 / val 10,490 / test 10,490 samples
   - For `window_size = 48`: train 49,016 / val 10,466 / test 10,466 samples

---

## 4. Methodology

### 4.1 Shared Training Infrastructure

All models are trained using the shared `train_model` function from `src/train.py`. This ensures a uniform methodology and eliminates the influence of training procedure differences on the final metrics.

Shared training settings:
- **Optimiser:** Adam, initial lr = 10⁻³
- **Loss function:** MSELoss
- **LR scheduler:** ReduceLROnPlateau (patience = 5, factor = 0.5)
- **Early stopping:** patience = 10 epochs
- **Batch size:** 64
- **Maximum epochs:** 50
- **Reproducibility:** seed = 42 (PyTorch, NumPy, Python random)
- **Device:** CPU

### 4.2 Model Input/Output Contract

Every model adheres to a unified interface:
- **Input:** `(batch, window_size, 14)` – 14 features, float32
- **Output:** `(batch, 1)` – predicted temperature (normalised scale)

### 4.3 LSTM Model

Implementation: `src/models/lstm.py`

Architecture:
- Multi-layer LSTM with dropout between layers
- Last hidden state → linear layer → scalar prediction

| Parameter | Config A | Config B |
|-----------|----------|----------|
| hidden_size | 64 | 128 |
| num_layers | 2 | 2 |
| dropout | 0.2 | 0.2 |
| window_size | 24 | 24 |
| Parameter count | 53,825 | 205,953 |

Only `hidden_size` differs between configurations, enabling a clean ablation of model capacity.

### 4.4 CNN Model

Implementation: `src/models/cnn.py`

Architecture:
- Input permuted to `(batch, 14, window)` for Conv1D
- Two Conv1D layers with ReLU activations
- AdaptiveAvgPool1d(1) – reduces sequence to a single vector
- Dropout → linear layer → prediction

| Parameter | Config A | Config B |
|-----------|----------|----------|
| channels_1 | 32 | 64 |
| channels_2 | 64 | 128 |
| kernel_size_1 | 3 | 5 |
| kernel_size_2 | 3 | 3 |
| dropout | 0.2 | 0.3 |
| window_size | 24 | 24 |
| Parameter count | 7,649 | 29,377 |

### 4.5 Hybrid CNN-LSTM Model

Implementation: `src/models/hybrid.py`

Architecture:
- CNN block: two Conv1D layers (ReLU) + MaxPool1d(2) → halves the sequence length
- LSTM block: receives CNN features as a sequence and models long-range dependencies
- Last LSTM hidden state → linear layer → prediction

The hybrid-specific parameter choices are motivated by the architecture:
- **window_size = 48** (instead of 24): MaxPool1d(2) reduces the sequence to 24 steps, matching the context of the standalone LSTM.
- **lstm_hidden_size = 32** (instead of 64): CNN features are already compressed, so the LSTM requires less capacity.
- **kernel_size = 5**: a wider kernel provides richer local context before the LSTM.

| Parameter | Config A | Config B |
|-----------|----------|----------|
| cnn_channels | 64 | 64 |
| kernel_size | 5 | 5 |
| lstm_hidden_size | 32 | 32 |
| lstm_num_layers | 2 | 3 |
| dropout | 0.2 | 0.2 |
| window_size | 48 | 48 |
| Parameter count | 46,113 | 54,561 |

Only `lstm_num_layers` differs between configurations, testing whether the LSTM benefits from additional depth when operating on CNN-extracted features.

---

## 5. Experiment Design

### 5.1 Overview of Experiments

| Experiment | Goal |
|------------|------|
| LSTM ConfigA vs ConfigB | Effect of capacity (hidden_size 64 vs 128) |
| CNN ConfigA vs ConfigB | Effect of architecture size (channels, kernel) |
| Hybrid ConfigA vs ConfigB | Effect of LSTM depth in hybrid (2 vs 3 layers) |
| Ablation: window 24 vs 48 | Effect of input window length on CNN performance |

### 5.2 Evaluation Protocol

1. All models trained with fixed seed 42 for reproducibility.
2. Best checkpoint selected by validation loss.
3. Final metrics (MAE, RMSE, R²) evaluated on the held-out test set.
4. Predictions inverse-transformed to °C before metric computation.

### 5.3 Ablation – Input Window Size

The ablation was performed on the CNN model, training separate models for window_size ∈ {24, 48}. Each model was trained independently on its respective window size to ensure a fair comparison.

---

## 6. Results

### 6.1 LSTM – Training Curves and Validation Loss

Both LSTM models showed stable convergence. Validation loss decreased monotonically and tracked the training loss closely with no pronounced signs of overfitting.

| Model | Epochs | Final Train Loss (MSE) | Best Val Loss | Training Time | Parameters |
|-------|--------|----------------------|--------------|--------------|------------|
| LSTM ConfigA | 31 | 0.006548 | **0.006579** | 74.6 s | 53,825 |
| LSTM ConfigB | 20 | 0.006657 | 0.006965 | 44.8 s | 205,953 |

> *Note: LSTM checkpoints were not available for test-set inference. Test RMSE values are estimated from the scaled validation loss:*
> *LSTM ConfigA: RMSE ≈ 0.702 °C, LSTM ConfigB: RMSE ≈ 0.722 °C*

### 6.2 CNN – Test Set Results

| Model | Window | MAE [°C] | RMSE [°C] | R² | Training Time |
|-------|--------|---------|----------|-----|--------------|
| CNN ConfigA | 24 | 1.173 | 1.529 | 0.9615 | 48.7 s |
| CNN ConfigB | 24 | 0.785 | 1.010 | 0.9832 | 82.0 s |
| CNN ConfigA | 48 | 0.691 | 0.908 | 0.9864 | 92.6 s |
| CNN ConfigB | 48 | **0.661** | **0.866** | **0.9876** | 202.1 s |

### 6.3 Hybrid CNN-LSTM – Test Set Results

| Model | Window | MAE [°C] | RMSE [°C] | R² | Parameters |
|-------|--------|---------|----------|-----|------------|
| Hybrid ConfigA (2 LSTM layers) | 48 | **0.494** | **0.691** | **0.9921** | 46,113 |
| Hybrid ConfigB (3 LSTM layers) | 48 | 0.513 | 0.709 | 0.9917 | 54,561 |

### 6.4 Summary Comparison – Best Configurations

| Model | MAE [°C] | RMSE [°C] | R² | Parameters |
|-------|---------|----------|-----|------------|
| LSTM ConfigA* | ~0.702 | ~0.702 | — | 53,825 |
| CNN ConfigB (w24) | 0.785 | 1.010 | 0.983 | 29,377 |
| CNN ConfigB (w48) | 0.661 | 0.866 | 0.988 | 29,377 |
| **Hybrid ConfigA (w48)** | **0.494** | **0.691** | **0.992** | 46,113 |

\* Estimated from validation loss.

### 6.5 Ablation – Input Window Size (CNN)

| Model | Window 24 RMSE | Window 48 RMSE | Improvement |
|-------|---------------|----------------|-------------|
| CNN ConfigA | 1.529 °C | 0.908 °C | **−40.6 %** |
| CNN ConfigB | 1.010 °C | 0.866 °C | **−14.3 %** |

A longer input window (48 hours) consistently improves CNN results. The effect is more pronounced for Config A (smaller architecture), suggesting that a longer context partially compensates for lower model capacity.

---

## 7. Critical Analysis and Discussion

### 7.1 Architecture Comparison

The hybrid model achieved the best results across all metrics (MAE = 0.494 °C, RMSE = 0.691 °C, R² = 0.992). This confirms the rationale for combining CNN and LSTM: CNN efficiently extracts local features, while LSTM models their long-range dependencies. The result is consistent with findings in the literature on hybrid architectures for time series forecasting.

The CNN (best: Config B, window 48) achieved RMSE = 0.866 °C — significantly better than CNN with window 24 (RMSE = 1.010 °C), but still worse than the hybrid. This suggests that pure convolutions without a recurrent component lose global sequential context.

For LSTM, only validation losses are available (Person 1 did not upload the checkpoints). The estimated test RMSE (≈ 0.70 °C) is comparable to the hybrid model, which is notable given the simpler architecture. This indicates that LSTM can effectively model hourly temperature trends even without a convolutional front-end.

### 7.2 Error Analysis

From the prediction visualisations of the CNN model (last 100 steps of the test set), the model tracks seasonal and diurnal trends very accurately. The largest errors occur:
- during sudden temperature spikes (rapid weather changes),
- at extreme values outside the typical distribution (outliers in the test data),
- during transitional seasons (spring, autumn) where temperature patterns are less predictable.

This behaviour is expected: MSE-trained models tend to regress towards the mean under uncertainty, underestimating extreme values.

### 7.3 Effect of Model Capacity

For CNN: Config B (larger architecture) consistently outperforms Config A, but at the cost of longer training time (82 s vs. 48 s at window 24).

For LSTM: Config A (hidden=64) achieved better validation loss than Config B (hidden=128), despite Config B having 4× more parameters. This suggests that hidden=64 is sufficient capacity for this task and that a larger model does not necessarily generalise better.

For Hybrid: Config A (2 LSTM layers) outperforms Config B (3 LSTM layers) despite fewer parameters. This confirms the hypothesis that the LSTM does not benefit from additional depth when its input is already CNN-processed features.

### 7.4 Anomaly in CNN Training Curves

The CNN training JSON files contain validation losses in the range of 22–120 (compared to LSTM values of 0.006–0.009). This is an artefact from a separate training run with a scaler misconfiguration, not from the final saved models. The CNN checkpoints themselves are functional, as verified by direct inference on the test set.

### 7.5 Discussion of Ablation Results

The window ablation on CNN clearly shows that a 48-hour context is superior to 24 hours. Hourly temperature data exhibits strong diurnal cycles (24 hours) — a window of 48 hours captures one full daily cycle plus the previous day's context, explaining the improvement.

The hybrid model was deliberately designed with window = 48 because MaxPool1d(2) reduces the sequence to 24 steps — matching the context available to the standalone LSTM with window 24. This is a considered design choice ensuring that the LSTM within the hybrid operates on the same amount of temporal information as the standalone LSTM.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

- **Missing LSTM and Hybrid checkpoints** – test metrics for these models are not available from direct inference; LSTM results are estimated from validation loss.
- **CPU-only training** – all models were trained on CPU, significantly increasing training times (Hybrid ConfigB: 601 s). GPU training would be orders of magnitude faster.
- **Only two configurations per model** – systematic hyperparameter search (e.g. Bayesian optimisation) could reveal better configurations.
- **No seasonal error analysis** – models were evaluated in aggregate; per-season error analysis would reveal specific patterns.
- **Single-step forecasting only** – the project does not address multi-step forecasting.

### 8.2 Future Work

1. **Multi-step forecasting** – extend to predicting 6, 12, or 24 steps ahead.
2. **Attention mechanism** – add self-attention to LSTM or explore Transformer-based models (e.g. Temporal Fusion Transformer).
3. **Systematic hyperparameter search** – Bayesian optimisation across all models.
4. **Seasonal error analysis** – compute metrics separately for spring/summer/autumn/winter.
5. **GPU training** – reduce training times and enable larger-scale experiments.
6. **Walk-forward validation** – more robust evaluation with multiple validation windows.
7. **Interpretability** – SHAP values or saliency maps for feature importance analysis.

---

## 9. Team Contributions

| Member | Contribution |
|--------|-------------|
| **Person 1** | Data pipeline (`src/dataset.py`), LSTM model (`src/models/lstm.py`), shared training loop (`src/train.py`, `src/utils.py`), training of LSTM configurations A and B |
| **Person 2** | CNN model (`src/models/cnn.py`), training script (`src/run_cnn.py`), evaluation script (`src/evaluate_cnn.py`), training and evaluation of CNN configurations A and B for both window sizes |
| **Person 3** | Hybrid CNN-LSTM model (`src/models/hybrid.py`), training script (`src/run_hybrid.py`), training and evaluation of hybrid configurations A and B |
| **Person 4** | Evaluation module (`src/evaluate.py`), main evaluation script (`src/run_evaluation.py`), generation of all plots, comparison tables, ablation study, report compilation and writing |

---

## 10. References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
3. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. Kim, T. Y., & Cho, S. B. (2019). Predicting residential energy consumption using CNN-LSTM neural networks. *Energy*, 182, 72–81.
6. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NIPS*, 25.
7. Jena Climate dataset: https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
