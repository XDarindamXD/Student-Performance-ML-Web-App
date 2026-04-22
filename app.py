import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit.components.v1 as components

# --- UI Setup ---
st.set_page_config(page_title="AI Student Predictor Pro", layout="wide")
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

    # --- 2. Preprocessing (Error Fix: String to Float) ---
    st.subheader("⚙️ Advanced Data Encoding")
    
    # pd.get_dummies automatically converts 'female', 'group A', etc. into 0s and 1s
    # Isse "ValueError: could not convert string to float" kabhi nahi aayega
    df_clean = pd.get_dummies(df, drop_first=True)
    
    st.write("Encoded Data (Dummies Created):")
    st.write(df_clean.head())

    # --- 3. Model Training Logic ---
    # Math score ko target maan rahe hain
    target_col = 'math score'
    
    if target_col in df_clean.columns:
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]

        # Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 4. Metrics Display
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("Model Accuracy (R² Score)", f"{r2*100:.2f}%")
        col2.metric("Mean Error (Marks)", f"{mae:.2f}")

        # 5. Visualizations
        st.subheader("📈 Model Performance & Feature Importance")
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            fig, ax = plt.subplots()
            sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
            plt.xlabel("Actual Marks")
            plt.ylabel("Predicted Marks")
            plt.title("Actual vs Predicted")
            st.pyplot(fig)

        with fig_col2:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='magma', ax=ax)
            plt.title('Top 10 Factors Influencing Marks')
            st.pyplot(fig)

        # 6. Risk Alert System
        st.subheader("🚨 Early Warning System (Sample Student Analysis)")
        student_idx = 0 # Testing first student from test set
        student_data = X_test.iloc[[student_idx]]
        actual = y_test.iloc[student_idx]
        pred = model.predict(student_data)[0]

        st.write(f"**Analysis for Student ID {student_idx}:**")
        if pred < 40:
            st.error(f"Predicted Score: {pred:.2f} | Actual: {actual} -> **CRITICAL RISK**")
        elif pred < 60:
            st.warning(f"Predicted Score: {pred:.2f} | Actual: {actual} -> **AVERAGE / AT RISK**")
        else:
            st.success(f"Predicted Score: {pred:.2f} | Actual: {actual} -> **GOOD PERFORMANCE**")

        # 7. SHAP (Explainable AI) - The "Khatarnak" Part
        st.subheader("🧠 Explainable AI: SHAP Value Interpretation")
        st.write("Ye chart dikhata hai ki model ne kis factor ko kitna weight diya:")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        def st_shap(plot):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=350)

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:]))

    else:
        st.error(f"Error: Dataset mein '{target_col}' column nahi mila! Check your CSV column names.")

else:
    st.info("👈 Please upload your 'StudentsPerformance.csv' file from the sidebar to begin.")
    st.stop()