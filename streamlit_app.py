import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="HR Attrition Prediction", layout="wide", page_icon="📊")

# Sidebar
st.sidebar.title("📁 Upload Employee CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
try:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
    st.dataframe(df.head())
except Exception as e:
    st.error("⚠️ Could not read the uploaded file. Please ensure it's a .csv file and not a renamed .txt or broken file.")
    st.markdown("📂 [Download Sample CSV from Google Drive](https://drive.google.com/your-link)")


# Load model
model = joblib.load("attrition_model.pkl")

st.title("🔍 Employee Attrition Prediction Dashboard")

# App description
st.markdown("""
Welcome to the **HR Analytics Dashboard**.  
This app predicts whether an employee will leave the company (attrition) using a machine learning model.
You can also explore key visual insights based on the uploaded data.
""")

# Sample CSV download
with open("WA_Fn-UseC_-HR-Employee-Attrition.csv", "rb") as f:
    st.download_button("📥 Download Sample CSV", f, "sample.csv", "text/csv")

# Process CSV
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show preview
    st.subheader("👁️ Preview of Uploaded Data")
    st.dataframe(df.head())

    # Encode target
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Heatmap checkbox
    if st.checkbox("📌 Show Heatmap"):
        st.subheader("📉 Correlation Heatmap")
        plt.figure(figsize=(16, 10))
        sns.heatmap(df_encoded.corr(), cmap='coolwarm')
        st.pyplot(plt)

    # Predictions
    if "Attrition" in df_encoded.columns:
        X = df_encoded.drop("Attrition", axis=1)
        predictions = model.predict(X)
        df["Predicted_Attrition"] = ["Yes" if pred == 1 else "No" for pred in predictions]
        st.subheader("🎯 Prediction Results")
        st.dataframe(df[["EmployeeNumber", "Attrition", "Predicted_Attrition"]].head())

    # --------- Graphs Section ----------
    st.subheader("📊 Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Attrition Count**")
        sns.countplot(data=df, x='Attrition')
        st.pyplot(plt.gcf())

        st.markdown("**Attrition by Department**")
        plt.figure()
        sns.countplot(data=df, x='Department', hue='Attrition')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

    with col2:
        st.markdown("**Monthly Income Distribution**")
        plt.figure()
        sns.histplot(data=df, x='MonthlyIncome', bins=30, kde=True)
        st.pyplot(plt.gcf())

        st.markdown("**Job Satisfaction vs Attrition**")
        plt.figure()
        sns.boxplot(data=df, x='Attrition', y='JobSatisfaction')
        st.pyplot(plt.gcf())
