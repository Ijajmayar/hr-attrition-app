import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("attrition_model.pkl")

# App Config
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>🔍 Employee Attrition Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Upload employee data and find out who is likely to leave 🔄</h4>", unsafe_allow_html=True)
st.write("---")

# Sidebar
st.sidebar.header("📁 Upload Employee CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Main Area
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Employee Data Preview")
    st.dataframe(df.head())

    # One-hot encode and align
    df_encoded = pd.get_dummies(df, drop_first=True)
    model_input = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    model_input = pd.get_dummies(model_input.drop("Attrition", axis=1), drop_first=True)
    df_encoded = df_encoded.reindex(columns=model_input.columns, fill_value=0)

    if st.button("🔮 Predict Attrition"):
        prediction = model.predict(df_encoded)
        df["Predicted Attrition"] = ["Yes" if val == 1 else "No" for val in prediction]

        st.subheader("📊 Prediction Results")
        for i, row in df.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"**Employee {row['EmployeeNumber']}**")
            with col2:
                if row["Predicted Attrition"] == "Yes":
                    st.warning("⚠️ At Risk of Leaving")
                else:
                    st.success("✅ Likely to Stay")
        st.write("---")

# Optional Correlation Heatmap
st.subheader("📈 Feature Correlation with Attrition")
if st.checkbox("Show Heatmap"):
    df_orig = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df_orig['Attrition'] = df_orig['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    corr = pd.get_dummies(df_orig, drop_first=True).corr()
    plt.figure(figsize=(10, 12))
    sns.heatmap(corr[['Attrition']].sort_values(by='Attrition', ascending=False), annot=True, cmap='coolwarm')
    st.pyplot(plt)
