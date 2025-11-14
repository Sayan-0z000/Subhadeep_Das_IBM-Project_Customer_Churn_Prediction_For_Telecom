import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# PAGE CONFIGURATION

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")
st.title("📡 Telecom Customer Churn Prediction System")
st.caption("Built with Machine Learning & Streamlit | by Subhadeep")
st.markdown("---")

# SAFE DATA LOADER FUNCTION

def load_data(uploaded_file):
    """Safely reads CSV or Excel files."""
    try:
        if uploaded_file.name.endswith('.csv'):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("⚠ Please upload a valid CSV or Excel file.")
            return None

        if df.empty or df.shape[1] == 0:
            st.error("⚠ The uploaded file is empty or invalid. Please upload a proper dataset.")
            return None

        st.success(f"✅ File loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None

# MAIN TABS

tab1, tab2, tab3, tab4 = st.tabs(["📁 Dataset", "⚙ Model Training", "🔮 Prediction", "📊 Insights"])
le = LabelEncoder()

# TAB 1: DATASET

with tab1:
    st.header("📁 Upload and Explore Dataset")

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.subheader("🔍 Dataset Preview")
            st.dataframe(df.head())

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

            st.write("### 📊 Summary Statistics")
            st.dataframe(df.describe())

            if 'Churn' in df.columns:
                st.write("### 📉 Churn Distribution")
                fig, ax = plt.subplots(figsize=(6, 2.5))
                sns.countplot(x='Churn', data=df, palette='Set2', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠ Dataset must include a 'Churn' column.")

# TAB 2: MODEL TRAINING (UPDATED)

with tab2:
    st.header("⚙ Train and Compare Multiple Models")

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            df = df.dropna()

            for col in df.select_dtypes(include=['object']).columns:
                df[col] = le.fit_transform(df[col])

            if 'Churn' not in df.columns:
                st.error("⚠ The dataset must have a 'Churn' target column.")
            else:
                X = df.drop('Churn', axis=1)
                y = df['Churn']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if st.button("🚀 Train All Models"):
                    st.info("Training Decision Tree, Random Forest, and XGBoost models...")

                    models = {
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    }

                    results = {}

                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        results[name] = acc
                        pickle.dump(model, open(f"{name.replace(' ', '_')}_model.pkl", 'wb'))

                    # Save training feature columns
                    pickle.dump(list(X_train.columns), open("model_columns.pkl", "wb"))

                    # Create results DataFrame
                    result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
                    st.success("✅ All Models Trained Successfully!")
                    st.dataframe(result_df)

                    # Plot model comparison
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    sns.barplot(x='Model', y='Accuracy', data=result_df, palette='coolwarm', ax=ax)
                    st.pyplot(fig)

                    best_model_name = max(results, key=results.get)
                    st.success(f"🏆 Best Model: *{best_model_name}* with Accuracy: {results[best_model_name]:.2f}")
                    st.session_state.best_model = f"{best_model_name.replace(' ', '_')}_model.pkl"

                    # Confusion matrix for best model
                    best_model = pickle.load(open(st.session_state.best_model, 'rb'))
                    y_pred_best = best_model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred_best)
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f"Confusion Matrix: {best_model_name}")
                    st.pyplot(fig)

                    # Add timestamp and best model info to results
                    from datetime import datetime
                    result_df["Trained_On"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    result_df.loc[len(result_df)] = [f"Best Model → {best_model_name}", results[best_model_name], datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

                    # Download CSV report
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Model Report (CSV)",
                        data=csv_data,
                        file_name="Model_Comparison_Report.csv",
                        mime="text/csv"
                    )
    else:
        st.warning("Please upload dataset in the 'Dataset' tab first.")

# TAB 3: PREDICTION

with tab3:
    st.header("🔮 Predict Churn for a Customer")

    model_choice = st.selectbox("Select Model", ["Decision_Tree_model.pkl", "Random_Forest_model.pkl", "XGBoost_model.pkl"])

    try:
        model = pickle.load(open(model_choice, 'rb'))
        st.success(f"✅ {model_choice} loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠ Train the models first in the previous tab.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col3:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)

        submitted = st.form_submit_button("🔍 Predict")

        if submitted and model is not None:
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [senior],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'Contract': [contract],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })

            # Encode categorical variables
            for col in input_data.select_dtypes(include=['object']).columns:
                input_data[col] = le.fit_transform(input_data[col])

            # Load training feature names
            try:
                model_columns = pickle.load(open("model_columns.pkl", "rb"))
            except FileNotFoundError:
                st.error("⚠ model_columns.pkl not found. Please retrain the model first.")
                st.stop()

            # Add any missing columns
            for col in model_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Ensure correct column order
            input_data = input_data[model_columns]

            # Make prediction
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("❌ The customer is *likely to churn*.")
            else:
                st.success("✅ The customer is *likely to stay*.")

# TAB 4: INSIGHTS

with tab4:
    st.header("📊 Insights and Visualizations")

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None and 'Churn' in df.columns:
            st.write("### Tenure vs Monthly Charges")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, palette='Set1', ax=ax)
            st.pyplot(fig)

            st.write("### Internet Service vs Churn")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set2', ax=ax)
            st.pyplot(fig)

            st.write("### Contract Type vs Churn")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            sns.countplot(x='Contract', hue='Churn', data=df, palette='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠ 'Churn' column not found in dataset.")
    else:
        st.info("Upload a dataset to view insights.")