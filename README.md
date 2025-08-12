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
├─ README.md
├─ .gitignore
├─ .env.example
├─ requirements.txt        # core deps shared across projects
├─ requirements-text.txt   # +nltk/spacy (only for NLP)
├─ requirements-vision.txt # +torch/torchvision or tensorflow (only for CV)
├─ Makefile                # convenience targets (see below)
├─ configs/
│  ├─ dataset.yaml         # paths, target, splits, modality
│  ├─ model.yaml           # model family & hyperparams
│  ├─ train.yaml           # epochs, batch_size, class_weights, seed
│  └─ eval.yaml            # metrics to compute/report
├─ data/
│  ├─ raw/                 # original data (not tracked by git)
│  ├─ interim/             # cleaned/standardized (e.g., tokenized text)
│  └─ processed/           # train/val/test ready for modeling
├─ models/
│  ├─ artifacts/           # .joblib / .pt / .h5 files
│  ├─ metrics/             # JSON/CSV metric logs per run
│  └─ version.txt          # pointer to "best" model
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_baseline.ipynb
│  └─ 03_experiments.ipynb
├─ reports/
│  ├─ figures/             # plots saved by EDA/eval
│  └─ model_card.md        # short, human-readable summary
├─ scripts/
│  ├─ download_data.sh     # optional data fetcher
│  ├─ run_train.sh
│  └─ run_eval.sh
├─ src/
│  ├─ __init__.py
│  ├─ cli.py               # `python -m src.cli <cmd> --config ...`
│  ├─ utils/
│  │  ├─ io.py             # load/save, path helpers
│  │  ├─ logging.py        # run id, console/file logs
│  │  └─ metrics.py        # RMSE/MAE/ROC AUC/F1, etc.
│  ├─ data/
│  │  ├─ load.py           # reads raw -> dataframe/paths
│  │  ├─ split.py          # stratified/temporal splits
│  │  └─ preprocess.py     # dispatches by modality
│  ├─ features/
│  │  ├─ tabular.py        # scaling, OHE/target enc, imputation
│  │  ├─ text.py           # clean, tokenize, TF-IDF/embeddings
│  │  └─ vision.py         # resize/normalize/augment
│  ├─ models/
│  │  ├─ tabular.py        # sklearn pipelines (LR/RF/XGB/LGBM)
│  │  ├─ text.py           # NB/LinearSVM + TF-IDF; optional deep
│  │  └─ vision.py         # CNN/transfer learning (Torch/TensorFlow)
│  ├─ training/
│  │  ├─ train.py          # fit loop; saves model+metrics
│  │  └─ evaluate.py       # loads model; computes metrics & plots
│  └─ inference/
│     └─ predict.py        # batch/CLI prediction entrypoint
└─ tests/
   ├─ test_data.py
   ├─ test_features.py
   └─ test_models.py
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

# Initialize a virtual environment via CLI
python -m src.cli init-venv --shell

# Run CLI commands
python -m src.cli prepare
```

## License

This project is licensed under the MIT License.
