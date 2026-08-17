import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix

# --- 1. Title and Description ---
st.title("Multi-Model Classification Engine: Telco Churn")
st.markdown("""
Upload your test data and select a machine learning model to evaluate its performance in predicting customer churn.
""")

# --- 2. Sidebar Controls ---
st.sidebar.header("User Inputs")

# a. Dataset upload option
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# b. Model selection dropdown
model_names = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
selected_model_name = st.sidebar.selectbox("Select ML Model", model_names)

# Dictionary to map the dropdown names to the saved .pkl filenames
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

# --- 3. Main Application Logic ---
if uploaded_file is not None:
    # Load the uploaded data
    test_data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", test_data.head())
    
    # Check if 'Churn' exists in the uploaded data
    if 'Churn' not in test_data.columns:
        st.error("Error: The uploaded CSV must contain a 'Churn' column for the target variable.")
    else:
        # Separate features (X) and target (y)
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        # Load the selected model
        model_path = os.path.join('model', model_files[selected_model_name])
        
        if not os.path.exists(model_path):
            st.error(f"Error: Model file {model_path} not found. Ensure models are saved in the 'model' directory.")
        else:
            model = joblib.load(model_path)
            
            # Make predictions
            y_pred = model.predict(X_test)
            # Handle probability predictions for AUC depending on the model
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = [0] * len(y_test)

            st.write("---")
            st.write(f"### Evaluating: **{selected_model_name}**")
            
            # --- c. Display of evaluation metrics ---
            st.write("#### Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col1.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
            
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col2.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            
            col3.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")
            col3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")
            
            st.write("---")
            
            # --- d. Confusion matrix or classification report ---
            colA, colB = st.columns(2)
            
            with colA:
                st.write("#### Classification Report")
                report = classification_report(y_test, y_pred)
                st.text(report)
                
            with colB:
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            xticklabels=['No Churn (0)', 'Churn (1)'], 
                            yticklabels=['No Churn (0)', 'Churn (1)'])
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
else:
    st.info("Please upload a test dataset (CSV) from the sidebar to begin.")
