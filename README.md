# Federated Learning for National Electricity Consumption Forecasting

## 1. Project Overview

This project implements a decentralized, privacy-preserving machine learning system to forecast the cumulative annual electricity consumption for the Netherlands. The core of this work is a **Federated Learning (FL)** algorithm that trains a global consensus model by learning from distributed, city-level datasets without ever centralizing the raw data.

The methodology and architecture were inspired by the concepts presented in the research paper: *N. Bastianello, A. I. Rikos and K. H. Johansson, "Online Distributed Learning with Quantized Finite-Time Coordination," IEEE Conference on Decision and Control, 2023.*

The final federated model demonstrated superior performance (**0.12% error**) compared to a powerful, centralized LightGBM benchmark model (**5.22% error**), validating the efficacy of the federated approach for this sparse and distributed dataset.

---

## 2. Methodology & Key Features

The project followed an iterative development process, detailed in the final Jupyter Notebook:

1.  **Centralized Benchmarking:** An initial **LightGBM Regressor** model was trained on the fully aggregated dataset to establish a performance baseline.

2.  **Federated Model Design:**
    *   **Architecture:** A decentralized system of city-based "agents" was designed.
    *   **Model:** A **Federated Ridge Regression** model was chosen for its stability and suitability for parameter averaging. L2 regularization was key to preventing overfitting on sparse local data.

3.  **Data-Driven Feature Forecasting:** A supporting system of simple **Linear Regression models** was built to predict a key input feature (`total_num_connections`) for future years, making the main forecast more robust and credible.

4.  **Advanced Feature Engineering:** The model's predictive power was significantly enhanced by creating time-series features like lags, year-over-year growth rates, interaction terms, and polynomial features.

5.  **Hyperparameter Optimization:** A systematic **Grid Search** was conducted to automatically find the optimal `learning_rate` and regularization strength (`alpha`), which was instrumental in minimizing the final forecast error.

---

## 3. Repository Structure
.
├── Electricity consumption Netherlands/
│ └── Electricity/
│ ├── cityname_electricity_2013.csv
│ ├── ...
│ └── (and other city data files)
├── Netherland's Electricity Federated Ridge Regression.ipynb
└── README.md
code
Code
-   **`Electricity consumption Netherlands/`**: Contains the raw, city-level electricity consumption data used for training and evaluation.
-   **`Netherland's Electricity Federated Ridge Regression.ipynb`**: The main Jupyter Notebook containing the entire end-to-end workflow, from data ingestion to the final 5-year forecast visualization.
-   **`README.md`**: This file.

---

## 4. Setup & Installation

To run this project, you need a Python environment with the following libraries installed. It is highly recommended to use a virtual environment.

```bash
pip install pandas numpy scikit-learn seaborn matplotlib tqdm jupyter
