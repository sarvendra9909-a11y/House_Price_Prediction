import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FIND PROJECT FOLDER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# LOAD MODEL COMPARISON RESULTS
# ============================================================

results_path = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)

results = pd.read_csv(results_path)


# ============================================================
# DISPLAY MODEL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HOUSE PRICE PREDICTION - MODEL COMPARISON")
print("=" * 70)

print()

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# FIND BEST MODELS
# ============================================================

best_rmse = results.loc[
    results["RMSE"].idxmin()
]

best_r2 = results.loc[
    results["R2"].idxmax()
]

best_adjusted_r2 = results.loc[
    results["Adjusted R2"].idxmax()
]


# ============================================================
# DISPLAY BEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST RESULTS")
print("=" * 70)

print(
    f"\nBest model according to RMSE: "
    f"{best_rmse['Model']}"
)

print(
    f"RMSE: ₹{best_rmse['RMSE']:,.2f}"
)

print(
    f"\nBest model according to R²: "
    f"{best_r2['Model']}"
)

print(
    f"R²: {best_r2['R2']:.4f}"
)

print(
    f"\nBest model according to Adjusted R²: "
    f"{best_adjusted_r2['Model']}"
)

print(
    f"Adjusted R²: "
    f"{best_adjusted_r2['Adjusted R2']:.4f}"
)


# ============================================================
# GRAPH 1 — RMSE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.title(
    "RMSE Comparison of House Price Prediction Models"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "RMSE"
)

plt.xticks(rotation=20)

plt.tight_layout()

rmse_path = os.path.join(
    BASE_DIR,
    "rmse_comparison.png"
)

plt.savefig(
    rmse_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2 — R²
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["R2"]
)

plt.title(
    "R² Comparison of House Price Prediction Models"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "R² Score"
)

plt.xticks(rotation=20)

plt.tight_layout()

r2_path = os.path.join(
    BASE_DIR,
    "r2_comparison.png"
)

plt.savefig(
    r2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3 — ADJUSTED R²
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["Adjusted R2"]
)

plt.title(
    "Adjusted R² Comparison of House Price Prediction Models"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Adjusted R² Score"
)

plt.xticks(rotation=20)

plt.tight_layout()

adjusted_r2_path = os.path.join(
    BASE_DIR,
    "adjusted_r2_comparison.png"
)

plt.savefig(
    adjusted_r2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)

print("\nThree graphs have been saved successfully:")

print("1. rmse_comparison.png")
print("2. r2_comparison.png")
print("3. adjusted_r2_comparison.png")

print("\nLocation:")
print(BASE_DIR)