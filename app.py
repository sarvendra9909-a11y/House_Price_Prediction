import streamlit as st
import pandas as pd
import joblib

# Professional CSS styling
st.markdown("""
<style>

    /* Main page */
    .stApp {
        background-color: #f4f7fb;
    }

    /* Main title */
    h1 {
        color: #17324d;
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Description */
    .stApp p {
        color: #526273;
        font-size: 16px;
    }

    /* Input labels */
    label {
        color: #17324d !important;
        font-weight: 600 !important;
    }

    /* Number input boxes */
    div[data-baseweb="input"] {
        background-color: white;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
    }

    /* Input text */
    input {
        color: #17324d !important;
        font-size: 16px !important;
    }

    /* Predict button */
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

/* Button hover */
.stButton > button:hover {
    background-color: #FF95A5 ;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Load the trained model
model = joblib.load("house_price_model.pkl")

r2_score_value = 0.5464
rmse_value = 1514173

# App title
st.title("🏠 House Price Prediction")
st.write(
    "This machine learning application predicts house prices "
    "using Multiple Linear Regression."
)

st.info(
    "Model: Multiple Linear Regression | "
    "Features: Area, Bedrooms, Bathrooms, Stories, Parking"
)

if "history" not in st.session_state:
    st.session_state.history = []

st.write("Enter the details of the house:")
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2_score_value:.2f}")

with col2:
    st.metric("RMSE", f"₹{rmse_value:,.0f}")


# Input fields
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        value=2000
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        value=3
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        value=2
    )

with col2:
    stories = st.number_input(
        "Number of Stories",
        min_value=1,
        value=2
    )

    parking = st.number_input(
        "Number of Parking Spaces",
        min_value=0,
        value=1
    )


# Prediction button
if st.button("🔮 Predict House Price", use_container_width=True):

    # Create input DataFrame
    new_house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "parking": [parking]
    })

    # Make prediction
    predicted_price = model.predict(new_house)[0]

    # Convert price into lakh
    price_lakh = predicted_price / 100000

    st.session_state.history.append({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Stories": stories,
    "Parking": parking,
    "Predicted Price": predicted_price
})

    # Display result
    st.success(
        f"Predicted House Price: ₹{predicted_price:,.2f}"
    )

    st.info(
        f"Approximately ₹{price_lakh:.2f} lakh"
    )

st.subheader("📊 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
else:
    st.write("No predictions made yet.")

if st.session_state.history:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()    
st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    "This application uses Multiple Linear Regression to predict "
    "house prices based on area, bedrooms, bathrooms, stories, "
    "and parking spaces."
)

st.divider()

st.caption("🏠 House Price Prediction | Built with Python, Scikit-learn & Streamlit")