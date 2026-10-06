import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #f4f7fb;
    }

    h1 {
        color: #17324d;
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    h2, h3 {
        color: #17324d;
    }

    .stApp p {
        color: #526273;
        font-size: 16px;
    }

    label {
        color: #17324d !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="input"] {
        background-color: white;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
    }

    input {
        color: #17324d !important;
        font-size: 16px !important;
    }

    div[data-baseweb="select"] {
        background-color: white;
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #FFB1B1;
        color: black;
        border: none;
        border-radius: 12px;
        padding: 14px;
        font-size: 17px;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    .stButton > button:hover {
        background-color: #FF95A5;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

linear_model = joblib.load(
    "models/linear_regression.pkl"
)

random_forest_model = joblib.load(
    "models/random_forest.pkl"
)

xgboost_model = joblib.load(
    "models/xgboost.pkl"
)

svr_model = joblib.load(
    "models/svr.pkl"
)


# ============================================================
# MODEL DICTIONARY
# ============================================================

models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model,
    "SVR": svr_model
}


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

results_df = pd.read_csv(
    "model_comparison.csv"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# TITLE
# ============================================================

st.title("🏠 House Price Prediction")

st.write(
    "A machine learning application for predicting house prices "
    "and comparing multiple regression algorithms."
)


st.info(
    "Models: Linear Regression, Random Forest, XGBoost & SVR | "
    "Features: Area, Bedrooms, Bathrooms, Stories & Parking"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("📊 Model Performance Comparison")

st.write(
    "Four regression algorithms were trained and evaluated "
    "using the same dataset and test data."
)


# Make display copy
display_df = results_df.copy()


# Round values
for column in [
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "Adjusted R2"
]:
    if column in display_df.columns:
        display_df[column] = display_df[column].round(4)


# Rename columns
display_df = display_df.rename(
    columns={
        "R2": "R²",
        "Adjusted R2": "Adjusted R²"
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_row = results_df.iloc[0]

best_model_name = best_model_row["Model"]


st.success(
    f"🏆 Best Performing Model: **{best_model_name}**"
)


# ============================================================
# BEST MODEL METRICS
# ============================================================

st.subheader("🏆 Best Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        "MAE",
        f"₹{best_model_row['MAE']:,.0f}"
    )


with col2:
    st.metric(
        "MSE",
        f"{best_model_row['MSE']:,.0f}"
    )


with col3:
    st.metric(
        "RMSE",
        f"₹{best_model_row['RMSE']:,.0f}"
    )


with col4:
    st.metric(
        "R²",
        f"{best_model_row['R2']:.4f}"
    )


with col5:
    st.metric(
        "Adjusted R²",
        f"{best_model_row['Adjusted R2']:.4f}"
    )


# ============================================================
# GRAPHICAL COMPARISON
# ============================================================

st.subheader("📈 Visual Model Comparison")


# Create graph data
chart_df = results_df.set_index("Model")


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

st.write("### RMSE Comparison")

st.caption(
    "Lower RMSE indicates better prediction performance."
)

st.bar_chart(
    chart_df["RMSE"]
)


# ------------------------------------------------------------
# R²
# ------------------------------------------------------------

st.write("### R² Comparison")

st.caption(
    "Higher R² indicates better explanatory performance."
)

st.bar_chart(
    chart_df["R2"]
)


# ------------------------------------------------------------
# Adjusted R²
# ------------------------------------------------------------

st.write("### Adjusted R² Comparison")

st.caption(
    "Higher Adjusted R² indicates better performance "
    "while accounting for the number of features."
)

st.bar_chart(
    chart_df["Adjusted R2"]
)


# ============================================================
# MODEL SELECTION
# ============================================================

st.subheader("🤖 Select Model for Prediction")

selected_model_name = st.selectbox(
    "Choose a machine learning model:",
    list(models.keys())
)


selected_model = models[selected_model_name]


# ============================================================
# SELECTED MODEL PERFORMANCE
# ============================================================

selected_row = results_df[
    results_df["Model"] == selected_model_name
].iloc[0]


st.write(
    f"### Performance of {selected_model_name}"
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        "MAE",
        f"₹{selected_row['MAE']:,.0f}"
    )


with col2:
    st.metric(
        "MSE",
        f"{selected_row['MSE']:,.0f}"
    )


with col3:
    st.metric(
        "RMSE",
        f"₹{selected_row['RMSE']:,.0f}"
    )


with col4:
    st.metric(
        "R²",
        f"{selected_row['R2']:.4f}"
    )


with col5:
    st.metric(
        "Adjusted R²",
        f"{selected_row['Adjusted R2']:.4f}"
    )


# ============================================================
# HOUSE INPUT
# ============================================================

st.subheader("🏡 Enter House Details")


col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        value=2000,
        step=100
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        value=2,
        step=1
    )


with col2:

    stories = st.number_input(
        "Number of Stories",
        min_value=1,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Number of Parking Spaces",
        min_value=0,
        value=1,
        step=1
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    new_house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "parking": [parking]
    })


    predicted_price = selected_model.predict(
        new_house
    )[0]


    price_lakh = predicted_price / 100000


    # Save history
    st.session_state.history.append({

        "Model": selected_model_name,

        "Area": area,

        "Bedrooms": bedrooms,

        "Bathrooms": bathrooms,

        "Stories": stories,

        "Parking": parking,

        "Predicted Price": predicted_price
    })


    st.success(
        f"🏠 Predicted House Price: "
        f"₹{predicted_price:,.2f}"
    )


    st.info(
        f"Approximately ₹{price_lakh:.2f} lakh"
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.subheader("📜 Prediction History")


if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )


    history_display = history_df.copy()

    history_display["Predicted Price"] = (
        history_display["Predicted Price"]
        .map(lambda x: f"₹{x:,.2f}")
    )


    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.write(
        "No predictions made yet."
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

if st.session_state.history:

    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()


# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.divider()

st.subheader("📌 Project Conclusion")

st.write(
    "Four regression algorithms were compared for house price "
    "prediction. Linear Regression achieved the best overall "
    "performance on this dataset."
)

st.write(
    f"Linear Regression achieved an R² score of "
    f"{best_model_row['R2']:.4f} and an Adjusted R² of "
    f"{best_model_row['Adjusted R2']:.4f}."
)

st.write(
    "The results demonstrate that a more complex machine "
    "learning algorithm does not always provide better "
    "performance. Model performance depends on the dataset "
    "and the relationship between its features and target."
)


# ============================================================
# METRIC EXPLANATION
# ============================================================

st.divider()

st.subheader("📚 Evaluation Metrics")

st.write(
    "**MAE:** Mean Absolute Error. Lower values are better."
)

st.write(
    "**MSE:** Mean Squared Error. Lower values are better."
)

st.write(
    "**RMSE:** Root Mean Squared Error. Lower values are better."
)

st.write(
    "**R²:** Measures how much variation in house prices "
    "is explained by the model. Higher values are better."
)

st.write(
    "**Adjusted R²:** Similar to R² but also considers the "
    "number of input features. Higher values are better."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 House Price Prediction | Python | Scikit-learn | "
    "XGBoost | Streamlit"
)