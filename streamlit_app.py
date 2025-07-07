import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io

st.set_page_config(page_title="HR Attrition Prediction", layout="wide", page_icon="📊")

# Sidebar
st.sidebar.title("📁 Upload Employee CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load model once
model = joblib.load("attrition_model.pkl")

st.title("🔍 Employee Attrition Prediction Dashboard")

st.markdown("""
Welcome to the **HR Analytics Dashboard**.  
This app predicts whether an employee will leave the company (attrition) using a machine learning model.
You can also explore key visual insights based on the uploaded data.
""")

# Sample CSV download
with open("WA_Fn-UseC_-HR-Employee-Attrition.csv", "rb") as f:
    st.download_button("📥 Download Sample CSV", f, "sample.csv", "text/csv")

# Safely process CSV
df = None  # define df outside try block

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue()
        if not content:
            st.error("🚫 The uploaded file is empty. Please select a valid CSV.")
        else:
            decoded_file = io.StringIO(content.decode("utf-8"))
            df = pd.read_csv(decoded_file)

            if df.shape[1] < 5:
                st.error("⚠️ The file has too few columns.")
                df = None
            else:
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())
    except pd.errors.EmptyDataError:
        st.error("❌ Could not read file. Try re-exporting it or use the sample CSV.")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")

# Process if data loaded
if df is not None:
    try:
        # Encode target column if exists
        if 'Attrition' in df.columns:
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

    except Exception as e:
        st.error(f"⚠️ Something went wrong during processing: {e}")
