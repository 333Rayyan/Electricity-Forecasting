# ⚡ Electricity Load Forecasting for Germany

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green.svg)

*A comprehensive machine learning project comparing ARMA, KRR, and LSTM models for predicting Germany's electricity load demand.*

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models Implemented](#-models-implemented)
- [Installation](#-installation)
- [Usage](#-usage)

## 🎯 Overview

This project implements and compares three different machine learning approaches for electricity load forecasting in Germany:

1. **AutoRegressive Moving Average (ARMA)** - Classical time series approach
2. **Kernel Ridge Regression (KRR)** - Feature-based machine learning approach  
3. **Long Short-Term Memory (LSTM)** - Deep learning approach with hyperparameter tuning

The goal is to accurately predict hourly electricity demand, which is crucial for:
- Grid stability and energy management
- Power plant scheduling and optimisation
- Energy trading and market operations
- Renewable energy integration planning

## 📊 Dataset

**Source**: ENTSO-E (European Network of Transmission System Operators) transparency platform

**Time Period**: 2015-2019 (excluding 2020 due to COVID-19 anomalies)

**Features**:
- `utc_timestamp`: Hourly timestamps
- `DE_load_actual_entsoe_transparency`: Actual electricity load (MW) - **Target variable**
- `DE_load_forecast_entsoe_transparency`: ENTSO-E official forecast (MW) - **Baseline comparison**

**Data Points**: ~43,800 hourly observations

### Data Preprocessing
- Removed 2020 data to avoid COVID-19 impact
- Handled missing values using mean imputation
- Extracted temporal features (hour, day of week, month) for KRR model
- Applied MinMax scaling for LSTM model

## 🤖 Models Implemented

### 1. ARMA Model (`notebooks/ARMA.ipynb`)
**Approach**: Classical time series forecasting
- **Order**: ARMA(2,2) based on ACF/PACF analysis, ARMA(5,4) based on AIC
- **Features**: Historical load values only
- **Strengths**: Simple, interpretable, computationally efficient
- **Use Case**: Baseline model for comparison

### 2. Kernel Ridge Regression (`notebooks/KRR.ipynb`)
**Approach**: Feature-based machine learning
- **Features**: Hour of day, day of week, month, ENTSO-E forecast
- **Kernels**: Linear, RBF, Polynomial
- **Preprocessing**: StandardScaler normalisation
- **Strengths**: Captures non-linear patterns, handles multiple features

### 3. LSTM Neural Network (`notebooks/LSTM.ipynb`)
**Approach**: Deep learning time series forecasting
- **Architecture**: Sequential LSTM layers with dropout
- **Sequence Length**: Optimised using Keras Tuner
- **Hyperparameter Tuning**: Automated optimisation of:
  - Number of LSTM units
  - Dropout rates
  - Learning rate
  - Batch size
- **Strengths**: Captures long-term dependencies and complex patterns

## 🏗️ Project Structure

```
Electricity-Forecasting/
├── 📁 data/
│   ├── time_series_60min_singleindex.csv    # Raw ENTSO-E data
│   └── germany_cleaned_load_data.csv        # Preprocessed data
├── 📁 notebooks/
│   ├── ARMA.ipynb                          # ARMA model implementation
│   ├── KRR.ipynb                           # Kernel Ridge Regression
│   ├── LSTM.ipynb                          # LSTM with hyperparameter tuning
│   ├── Forecast.ipynb                      # ENTSO-E baseline analysis
│   └── Compare.ipynb                       # Model comparison
├── 📁 utils/
│   └── cleaning.py                         # Data preprocessing script
├── requirements.txt                       # Python dependencies
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.12+

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/333Rayyan/Electricity-Forecasting.git
cd Electricity-Forecasting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Data Preprocessing
```bash
python utils/cleaning.py
```
This will:
- Load raw ENTSO-E data
- Filter for Germany-specific columns
- Remove 2020 data
- Handle missing values
- Save cleaned data to `data/germany_cleaned_load_data.csv`

### 2. Run Individual Models

**ARMA Model:**
```bash
jupyter notebook notebooks/ARMA.ipynb
```

**Kernel Ridge Regression:**
```bash
jupyter notebook notebooks/KRR.ipynb
```

**LSTM with Hyperparameter Tuning:**
```bash
jupyter notebook notebooks/LSTM.ipynb
```
### 3. Analyse ENTSO-E Baseline
```bash
jupyter notebook notebooks/Forecast.ipynb
```

### 4. Compare Models
```bash
jupyter notebook notebooks/Compare.ipynb
```
