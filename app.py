import pandas as pd
import joblib
import streamlit as st

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    cat_columns = joblib.load("cat_columns.pkl")
    num_columns = joblib.load("num_columns.pkl")
    return model, scaler, encoders, feature_columns, cat_columns, num_columns


model, scaler, encoders, feature_columns, cat_columns, num_columns = load_artifacts()

st.set_page_config(page_title="Gaming Addiction Predictor")
st.title("Expect gaming addiction")

user_input = {}

with st.form("prediction_form"):
    st.subheader("Basic data")

    cols_per_row = 2
    all_cols = feature_columns
    for i in range(0, len(all_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(all_cols[i:i + cols_per_row]):
            with row_cols[j]:
                if col_name in cat_columns:
                    options = list(encoders[col_name].classes_)
                    user_input[col_name] = st.selectbox(col_name, options)
                else:
                    user_input[col_name] = st.number_input(
                        col_name, value=0.0, step=1.0
                    )

    submitted = st.form_submit_button("predect")


if submitted:
    input_df = pd.DataFrame([user_input])[feature_columns]

    for col in cat_columns:
        input_df[col] = encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("result")
    if prediction == 1:
        st.error("Addicted")
    else:
        st.success("Not Addicted")