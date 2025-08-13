# Car Price Prediction

Predict the selling price of used cars using regression models. This project explores a public Kaggle dataset containing features such as car brand, year, mileage, fuel type, transmission type, and more. The goal is to build an accurate predictive model that helps estimate fair market value.

## Dataset

**Source:** [Car Price Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/car-price-prediction-dataset)

Contains various attributes:
- Car_Name
- Year
- Selling_Price
- Present_Price
- Kms_Driven
- Fuel_Type
- Transmission
- Owner

## Codebase Structure

```
Car-Price-Prediction-Project/
├── README.md
├── app_streamlit_full.py
├── configs/
│   ├── dataset.yaml
│   ├── eval.yaml
│   ├── model.yaml
│   └── train.yaml
├── requirements.txt
└── src/
    ├── cli.py
    ├── data/
    │   ├── load.py
    │   ├── preprocess.py
    │   └── split.py
    ├── features/
    │   └── tabular.py
    ├── inference/
    │   └── predict.py
    ├── models/
    │   └── tabular.py
    └── training/
        ├── evaluate.py
        └── train.py
```

## Project Workflow

1. **Data Loading & Cleaning**
   - Handle missing values and incorrect data types
   - Normalize/encode categorical variables

2. **Exploratory Data Analysis (EDA)**
   - Visualize price trends based on car features
   - Identify correlations between predictors and price

3. **Feature Engineering**
   - Create new variables (e.g., car age)
   - Encode categorical variables (One-Hot Encoding / Label Encoding)

4. **Modeling**
   - Supports Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Random Forest, XGBoost, and LightGBM
   - Hyperparameter tuning for best performance

5. **Evaluation**
   - Metrics: R², MAE, RMSE
   - Compare models and select the most accurate one

## Tools & Libraries

- **Python 3.x** – Core programming language for data processing and modeling
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computing and array operations
- **Scikit-learn** – Machine learning models (Linear, Ridge, Lasso, Elastic Net, SVR) and utilities
- **XGBoost** – Gradient boosting regression model for structured/tabular data
- **Joblib** – Model persistence and artifact storage
- **Streamlit** – Interactive web-based interface for running the full ML pipeline

## How to Run

```bash
# Clone the repository
git clone https://github.com/InnerWerks/Car-Price-Prediction-Project.git

# Navigate into the project directory
cd car-price-prediction

# Run the app
make gui
```

## License

This project is licensed under the MIT License.
