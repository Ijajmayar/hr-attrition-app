import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

st.set_page_config(page_title="HR Attrition Prediction", layout="wide", page_icon="📊")

try:
    # Sidebar
    st.sidebar.title("📁 Upload Employee CSV Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    # Model selection
    model_option = st.sidebar.selectbox("🔍 Choose ML Model", ["Random Forest", "Logistic Regression", "XGBoost"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Encode target
        if 'Attrition' not in df.columns:
            st.error("❌ 'Attrition' column not found in uploaded CSV.")
            st.stop()

        df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
        df_encoded = pd.get_dummies(df, drop_first=True)

        if 'Attrition' not in df_encoded.columns:
            st.error("❌ 'Attrition' column missing after encoding. Check your data.")
            st.stop()

        # Split features and target
        X = df_encoded.drop("Attrition", axis=1)
        y = df_encoded["Attrition"]

        # Model setup
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)
        predictions = model.predict(X)
        df["Predicted_Attrition"] = ["Yes" if p == 1 else "No" for p in predictions]

        st.subheader("🎯 Prediction Results")
        st.dataframe(df[["EmployeeNumber", "Attrition", "Predicted_Attrition"]].head())

        # Evaluation Metrics
        st.subheader("📉 Evaluation Metrics")
        cm = confusion_matrix(y, predictions)
        st.text("Confusion Matrix:")
        st.write(cm)
        st.text("Classification Report:")
        st.text(classification_report(y, predictions))

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid()
            st.pyplot(plt)

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("📌 Feature Importances")
            importances = pd.Series(model.feature_importances_, index=X.columns)
            fig, ax = plt.subplots(figsize=(10, 8))
            importances.nlargest(15).plot(kind='barh', ax=ax)
            plt.title('Top 15 Feature Importances')
            st.pyplot(fig)

        # Visualization
        st.subheader("📊 Data Insights")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Attrition Count**")
            sns.countplot(data=df, x='Attrition')
            st.pyplot(plt.gcf())

            if 'Department' in df.columns:
                st.markdown("**Attrition by Department**")
                plt.figure()
                sns.countplot(data=df, x='Department', hue='Attrition')
                plt.xticks(rotation=45)
                st.pyplot(plt.gcf())

        with col2:
            if 'MonthlyIncome' in df.columns:
                st.markdown("**Monthly Income Distribution**")
                plt.figure()
                sns.histplot(data=df, x='MonthlyIncome', bins=30, kde=True)
                st.pyplot(plt.gcf())

            if 'JobSatisfaction' in df.columns:
                st.markdown("**Job Satisfaction vs Attrition**")
                plt.figure()
                sns.boxplot(data=df, x='Attrition', y='JobSatisfaction')
                st.pyplot(plt.gcf())

except Exception as e:
    st.error(f"🚫 An unexpected error occurred:\n{e}")
