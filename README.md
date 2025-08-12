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
   - Train baseline regression models (Linear, Ridge, Lasso)  
   - Experiment with tree-based models (Random Forest, XGBoost, LightGBM)  
   - Hyperparameter tuning for best performance  

5. **Evaluation**  
   - Metrics: R², MAE, RMSE  
   - Compare models and select the most accurate one  

## Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost, LightGBM (optional)
- Jupyter Notebook

## Results

The final model provides a strong predictive capability for used car prices, making it useful for dealers, buyers, and sellers to estimate fair market value.

## How to Run

```bash
# Clone the repository
git clone https://github.com/InnerWerks/Car-Price-Prediction-Project.git

# Navigate into the project directory
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook CarPricePrediction.ipynb
```

## License

This project is licensed under the MIT License.
