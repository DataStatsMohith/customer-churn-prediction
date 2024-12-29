import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import json

# Visualization libraries (used in Batch Prediction and Dashboard tabs)
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Numerical operations (if needed in batch or dashboard calculations)
import numpy as np

# Function to load Lottie animation from a local file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animations from local files
animations = {
    "upload": load_lottiefile("animations/upload.json"),
    "dashboard": load_lottiefile("animations/dashboard.json"),
    "charts": load_lottiefile("animations/ChartsAnimation.json"),
    "success": load_lottiefile("animations/success.json"),
    "churn": load_lottiefile("animations/leaving.json"),
    "head": load_lottiefile("animations/headanimation.json"),
}

# Load pre-trained model and artifacts
model = joblib.load("final_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")
optimal_threshold = joblib.load("optimal_threshold.pkl")["optimal_threshold"]

# Function to preprocess input data
def preprocess_input(data):
    # Handle empty strings in numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in data.columns:
            # Replace empty strings with NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill NaN values with the median
            data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in data.columns:
            # Ensure encoder.classes_ is a NumPy array
            if isinstance(encoder.classes_, list):
                encoder.classes_ = np.array(encoder.classes_)
            # Map values using the encoder
            data[col] = data[col].map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
            # Handle unseen categories
            if -1 not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, -1)

    # Scale numerical columns
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    # Ensure all necessary feature columns exist
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default value

    # Align the input data with the model's feature columns
    return data[feature_columns]


# Streamlit App
st.title("📊 Customer Churn Prediction")
st_lottie(animations["head"], height=200, key="welcome_animation")
st.subheader("Make data-driven decisions with insights into customer churn.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Dashboard"])

# Initialize session state
if "batch_data" not in st.session_state:
    st.session_state["batch_data"] = None
if "predictions_done" not in st.session_state:
    st.session_state["predictions_done"] = False

# Single Prediction Tab
with tab1:
    st.header("🧑‍💻 Single Customer Prediction")

    # Collect user inputs
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        tenure = st.slider("Tenure (months)", 0, 72, key="tenure")
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, key="monthly_charges")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, key="total_charges")

    contract = st.radio(
        "Contract Type", ["Month-to-month", "One year", "Two year"], horizontal=True, key="contract"
    )

    # Submit button
    if st.button("Submit", key="submit_button"):
        with st.spinner("Processing..."):
            # Prepare input data
            input_data = pd.DataFrame({
                'gender': [gender],
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract]
            })

            # Preprocess the input
            try:
                processed_data = preprocess_input(input_data)

                # Make predictions
                probabilities = model.predict_proba(processed_data)[0]
                prediction = (probabilities[1] >= optimal_threshold).astype(int)

                # Display result
                if prediction == 1:
                    st.error(f"Prediction: Churn (Confidence: {probabilities[1] * 100:.2f}%)")
                    st_lottie(animations["churn"], height=150, key="churn_animation")
                else:
                    st.success(f"Prediction: No Churn (Confidence: {probabilities[0] * 100:.2f}%)")
                    st_lottie(animations["success"], height=150, key="success_animation")
            except Exception as e:
                st.error(f"Error during processing: {e}")


# Batch Prediction Tab
with tab2:
    st.header("📂 Batch Prediction")
    st_lottie(animations["charts"], height=150, key="charts_animation")

    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")

    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        st.session_state["batch_data"] = batch_data
        st.session_state["predictions_done"] = False
        st.session_state["progress_percentage"] = 0
        st.session_state["processing"] = True

        # Initialize progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        chunk_size = 1000  # Number of rows to process per chunk
        total_rows = len(batch_data)
        processed_rows = 0

        with st.spinner("Processing batch predictions..."):
            predictions = []

            # Process the data in chunks
            for start_row in range(0, total_rows, chunk_size):
                end_row = min(start_row + chunk_size, total_rows)
                chunk = batch_data.iloc[start_row:end_row]

                # Preprocess the chunk and make predictions
                processed_chunk = preprocess_input(chunk)
                probabilities = model.predict_proba(processed_chunk)
                chunk_predictions = (probabilities[:, 1] >= optimal_threshold).astype(int)
                predictions.extend(chunk_predictions)

                # Update progress
                processed_rows += len(chunk)
                progress_percentage = int((processed_rows / total_rows) * 100)
                progress_bar.progress(progress_percentage)
                progress_text.text(f"Processed {processed_rows}/{total_rows} rows...")

            # Add predictions back to the DataFrame
            batch_data['Churn Prediction'] = predictions
            st.session_state["batch_data"] = batch_data
            st.session_state["predictions_done"] = True
            st.session_state["processing"] = False

        st.success("Batch predictions completed!")
        st.write(st.session_state["batch_data"])

        # Add a download button for the predictions
        st.download_button(
            "Download Predictions",
            st.session_state["batch_data"].to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

# Dashboard Tab
with tab3:
    st.header("📊 Dashboard")

    if st.session_state["batch_data"] is None:
        st.warning("Upload a CSV in the batch predictions tab to view dashboard insights.")
        st_lottie(animations["upload"], height=150, key="dashboard_no_data")
    else:
        if st.session_state.get("processing", False):  # Check if processing is ongoing
            with st.spinner("Batch predictions are being generated. Please wait..."):
                st_lottie(animations["upload"], height=150, key="loading_animation")
                st.progress(st.session_state["progress_percentage"])  # Show the shared progress bar
        else:
            st.lottie(animations["dashboard"], height=200, key="dashboard_animation")

            st.subheader("📈 Churn Prediction Visualizations")

            # Filters
            filter_gender = st.selectbox(
                "Filter by Gender",
                ['All'] + st.session_state["batch_data"]['gender'].unique().tolist(),
                key="filter_gender_dashboard"
            )
            filter_contract = st.selectbox(
                "Filter by Contract Type",
                ['All'] + st.session_state["batch_data"]['Contract'].unique().tolist(),
                key="filter_contract_dashboard"
            )

            # Apply filters
            filtered_data = st.session_state["batch_data"].copy()
            if filter_gender != 'All':
                filtered_data = filtered_data[filtered_data['gender'] == filter_gender]
            if filter_contract != 'All':
                filtered_data = filtered_data[filtered_data['Contract'] == filter_contract]

            # Key Performance Indicators (KPIs)
            st.header("📊 Key Performance Indicators (KPIs)")
            total_customers = len(filtered_data)
            churned_customers = (filtered_data['Churn Prediction'] == 1).sum()
            churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", total_customers)
            col2.metric("Churned Customers", churned_customers)
            col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")
            st.caption(f"The churn rate is approximately {churn_rate:.2f}%, with {churned_customers} customers predicted to churn out of {total_customers} total customers.")

            # Churn Prediction Distribution Pie Chart
            churn_counts = filtered_data['Churn Prediction'].value_counts().reset_index()
            churn_counts.columns = ['Churn Prediction', 'Count']
            churn_counts['Churn Prediction'] = churn_counts['Churn Prediction'].replace({0: "No Churn", 1: "Churn"})
            fig1 = px.pie(
                churn_counts,
                values='Count',
                names='Churn Prediction',
                title="Churn Prediction Distribution",
                color='Churn Prediction',
                color_discrete_map={"No Churn": "green", "Churn": "red"}
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(f"The churn rate is approximately {churn_rate:.2f}%, highlighting that {churned_customers} customers out of {total_customers} are predicted to churn.")

            # Additional Visualizations
            st.subheader("📈 Additional Insights")

            # Churn Rate by Gender
            gender_churn_rate = (
                filtered_data.groupby("gender")["Churn Prediction"]
                .mean()
                .reset_index()
                .rename(columns={"Churn Prediction": "Churn Rate"})
            )
            fig2 = px.bar(
                gender_churn_rate,
                x="gender",
                y="Churn Rate",
                title="Churn Rate by Gender",
                labels={"Churn Rate": "Churn Rate (%)"},
                color="gender",
                text="Churn Rate"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Female customers have a churn rate of {gender_churn_rate.loc[gender_churn_rate['gender'] == 'Female', 'Churn Rate'].values[0]:.2f}%, while male customers have a churn rate of {gender_churn_rate.loc[gender_churn_rate['gender'] == 'Male', 'Churn Rate'].values[0]:.2f}%. This insight can help in gender-specific churn mitigation strategies.")

            # Churn Rate by Contract Type
            contract_churn_rate = (
                filtered_data.groupby("Contract")["Churn Prediction"]
                .mean()
                .reset_index()
                .rename(columns={"Churn Prediction": "Churn Rate"})
            )
            fig3 = px.bar(
                contract_churn_rate,
                x="Contract",
                y="Churn Rate",
                title="Churn Rate by Contract Type",
                labels={"Churn Rate": "Churn Rate (%)"},
                color="Contract",
                text="Churn Rate"
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(f"'Month-to-month' contracts have the highest churn rate of {contract_churn_rate.loc[contract_churn_rate['Contract'] == 'Month-to-month', 'Churn Rate'].values[0]:.2f}%. This indicates potential issues with short-term contracts and opportunities for improvement.")

            # Churn by Tenure
            tenure_churn_rate = (
                filtered_data.groupby("tenure")["Churn Prediction"]
                .mean()
                .reset_index()
                .rename(columns={"Churn Prediction": "Churn Rate"})
            )
            fig4 = px.line(
                tenure_churn_rate,
                x="tenure",
                y="Churn Rate",
                title="Churn Rate by Tenure",
                labels={"Churn Rate": "Churn Rate (%)", "tenure": "Tenure (Months)"},
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Churn rates decrease steadily with increasing tenure. Customers with longer tenures are less likely to churn, highlighting the value of retaining customers over time.")

            # Monthly Charges Distribution
            fig5 = px.histogram(
                filtered_data,
                x="MonthlyCharges",
                color="Churn Prediction",
                barmode="overlay",
                title="Monthly Charges Distribution by Churn",
                labels={"MonthlyCharges": "Monthly Charges ($)", "Churn Prediction": "Churn Status"},
                color_discrete_map={0: "green", 1: "red"},
            )
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("Churned customers tend to have higher monthly charges compared to non-churned customers. This insight suggests a need to review pricing strategies for high-value customers.")

            # Correlation Heatmap
            st.subheader("📊 Feature Correlations")
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Churn Prediction"]
            cleaned_data = filtered_data[numerical_columns].copy()
            for col in numerical_columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors="coerce").fillna(0)
            correlation_matrix = cleaned_data.corr()
            fig6, ax = plt.subplots()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig6, clear_figure=True)
            st.caption("Key insights: 'Tenure' has a negative correlation with churn, while 'MonthlyCharges' shows a moderate positive correlation. These factors can guide targeted retention strategies.")
