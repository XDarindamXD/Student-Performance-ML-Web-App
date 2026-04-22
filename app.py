import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit.components.v1 as components

# --- UI Setup ---
st.set_page_config(page_title="AI Student Predictor", layout="wide")
st.title("🚀 Student Performance Prediction Dashboard")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your Student CSV File", type=["csv"])

if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    
    # Show Data Preview
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # 2. Preprocessing
    # Copy banayein taaki original data disturb na ho
    df_clean = df.copy()
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = le.fit_transform(df_clean[col])

    # 3. Model Training Logic
    # Hum assume kar rahe hain 'math score' target hai
    if 'math score' in df_clean.columns:
        X = df_clean.drop('math score', axis=1)
        y = df_clean['math score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 4. Metrics Display
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("Accuracy (R2 Score)", f"{r2*100:.2f}%")
        col2.metric("Mean Absolute Error", f"{mae:.2f} marks")

        # 5. Visualizations
        st.subheader("📈 Model Performance & Feature Importance")
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            fig, ax = plt.subplots()
            sns.regplot(x=y_test, y=y_pred, ci=None, color="blue", ax=ax)
            plt.title("Actual vs Predicted Marks")
            st.pyplot(fig)

        with fig_col2:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
            plt.title('Factors Affecting Math Scores')
            st.pyplot(fig)

        # 6. Risk Alert System (Khatarnak Update)
        st.subheader("🚨 Student Risk Analysis (Sample #5)")
        student_idx = 5
        student_data = X_test.iloc[[student_idx]]
        actual = y_test.iloc[student_idx]
        pred = model.predict(student_data)[0]

        if pred < 40:
            st.error(f"CRITICAL: Student predicted to score {pred:.2f} (Actual: {actual}). High Risk!")
        elif pred < 60:
            st.warning(f"WARNING: Student predicted to score {pred:.2f} (Actual: {actual}). Below Average.")
        else:
            st.success(f"SAFE: Student predicted to score {pred:.2f} (Actual: {actual}). Performing Well.")

        # 7. SHAP (Explainable AI)
        st.subheader("🧠 Why this prediction? (SHAP Analysis)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        def st_shap(plot):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=400)

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:]))

    else:
        st.error("Dataset mein 'math score' column nahi mila. Please check your CSV.")

else:
    st.info("👈 Please upload the 'StudentsPerformance.csv' file from the sidebar to begin.")
    st.stop()