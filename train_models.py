# ============================================================
# HOUSE PRICE PREDICTION - MODEL COMPARISON
# Linear Regression vs Random Forest vs XGBoost vs SVR
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data_path = "data/Housing_Price_Dataset.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

# Features used by our Streamlit application
features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

target = "price"


# Check whether required columns exist
required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR!")
    print("The following columns are missing from the dataset:")
    print(missing_columns)

    print("\nAvailable columns are:")
    print(df.columns.tolist())

    raise ValueError(
        "Please check the column names in your dataset."
    )


# Select only required columns
df = df[required_columns]


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("\nChecking missing values:")

print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

print("\nDataset shape after removing missing values:")
print(df.shape)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df[features]
y = df[target]

print("\nFeatures used:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 6. CREATE MODELS
# ============================================================

# ------------------------------------------------------------
# Model 1: Multiple Linear Regression
# ------------------------------------------------------------

linear_model = LinearRegression()


# ------------------------------------------------------------
# Model 2: Random Forest
# ------------------------------------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1
)


# ------------------------------------------------------------
# Model 3: XGBoost
# ------------------------------------------------------------

xgboost_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# Model 4: Support Vector Regression
# ------------------------------------------------------------

# SVR works better when features are scaled.
# Pipeline performs scaling automatically before SVR.

svr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(
        kernel="rbf",
        C=1000,
        gamma="scale",
        epsilon=0.1
    ))
])


# ============================================================
# 7. STORE ALL MODELS
# ============================================================

models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model,
    "SVR": svr_model
}


# ============================================================
# 8. CREATE FUNCTION FOR ADJUSTED R²
# ============================================================

def adjusted_r2_score(r2, n, p):
    """
    Calculate Adjusted R².

    r2 = R² score
    n  = number of observations
    p  = number of predictors/features
    """

    if n - p - 1 <= 0:
        return np.nan

    adjusted_r2 = (
        1 - ((1 - r2) * (n - 1) / (n - p - 1))
    )

    return adjusted_r2


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

print("\n")
print("=" * 70)
print("MODEL TRAINING STARTED")
print("=" * 70)


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    # Number of observations
    n = len(y_test)

    # Number of predictors
    p = X_test.shape[1]

    adjusted_r2 = adjusted_r2_score(
        r2,
        n,
        p
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adjusted_r2
    })

    print(f"{model_name} completed!")


# ============================================================
# 10. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)


# Sort according to RMSE
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved as:")
print("model_comparison.csv")


# ============================================================
# 13. SAVE EACH MODEL
# ============================================================

# Create models folder
os.makedirs("models", exist_ok=True)


for model_name, model in models.items():

    # Convert model name into a filename
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    filepath = os.path.join(
        "models",
        filename
    )

    joblib.dump(
        model,
        filepath
    )

    print(f"Saved: {filepath}")


# ============================================================
# 14. FIND BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model:", best_model_name)

print(
    "RMSE:",
    f"{results_df.iloc[0]['RMSE']:.2f}"
)

print(
    "R²:",
    f"{results_df.iloc[0]['R2']:.4f}"
)

print(
    "Adjusted R²:",
    f"{results_df.iloc[0]['Adjusted R2']:.4f}"
)


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

# This keeps compatibility with your existing app.py
joblib.dump(
    best_model,
    "house_price_model.pkl"
)

print("\nBest model saved as:")
print("house_price_model.pkl")


# ============================================================
# 16. SAVE MODEL INFORMATION
# ============================================================

model_information = {
    "best_model": best_model_name,
    "features": features,
    "target": target
}

joblib.dump(
    model_information,
    "model_information.pkl"
)

print("Model information saved as:")
print("model_information.pkl")


# ============================================================
# 17. FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("=" * 70)

print("\nFiles created:")

print("1. model_comparison.csv")
print("2. models/linear_regression.pkl")
print("3. models/random_forest.pkl")
print("4. models/xgboost.pkl")
print("5. models/svr.pkl")
print("6. house_price_model.pkl")
print("7. model_information.pkl")

print("\nNext step:")
print("Update app.py to display the model comparison.")