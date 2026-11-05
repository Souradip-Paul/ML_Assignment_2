
# Import libraries required for the Streamlit application.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# Configure the Streamlit page.
st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)


# Define the project and model directories.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# Define the original input features used during training.
FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


# Define the five saved model filenames.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


# Load the saved models and fitted preprocessing object.
@st.cache_resource
def load_artifacts():

    # Load the fitted preprocessor.
    loaded_preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.joblib"
    )

    # Load all five trained classification models.
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_preprocessor, loaded_models


# Convert the uploaded target column into binary values.
def prepare_target(target):

    # Make a copy so that the uploaded DataFrame is not modified.
    target = target.copy()

    # Handle datasets containing textual target labels.
    if target.dtype == "object":
        target = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .replace({"no": 0, "yes": 1})
        )

    # Convert the target values into numeric form.
    target = pd.to_numeric(target, errors="coerce")

    # Reject missing or invalid target values.
    if target.isna().any():
        raise ValueError(
            "The target column 'y' contains missing or invalid values."
        )

    # Ensure that the target contains only binary labels.
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError(
            "The target column 'y' must contain only 0/1 or no/yes values."
        )

    return target.astype(int)


# Generate prediction scores for ROC-AUC calculation.
def get_prediction_scores(model, X_processed):

    # Use predicted probabilities when the model supports them.
    if hasattr(model, "predict_proba"):

        return model.predict_proba(X_processed)[:, 1]

    # Otherwise, use the model's decision scores.
    return model.decision_function(X_processed)


# Calculate all six evaluation metrics.
def evaluate_model(model, X_processed, y_true):

    # Generate class predictions.
    y_pred = model.predict(X_processed)

    # Generate continuous scores for ROC-AUC.
    y_scores = get_prediction_scores(model, X_processed)

    # Calculate ROC-AUC when both classes are present.
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_scores)
    else:
        auc = np.nan

    # Calculate the evaluation metrics.
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    return results, y_pred


# Display the application heading and introduction.
st.title("Bank Marketing Classification")

st.write(
    "This application evaluates five trained machine learning "
    "classification models on an uploaded Bank Marketing test dataset. "
    "The saved models and preprocessing object are loaded directly; "
    "no model retraining takes place."
)


# Load the saved models and preprocessing object.
try:

    preprocessor, trained_models = load_artifacts()

except Exception as error:

    st.error(
        f"Unable to load the saved model files. "
        f"Please check the model directory. Details: {error}"
    )

    st.stop()


# Allow the user to upload a test CSV file.
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 15 input features and target column y.",
    type=["csv"]
)


# Continue only after a CSV file has been uploaded.
if uploaded_file is not None:

    try:

        # Read the uploaded CSV file.
        test_data = pd.read_csv(uploaded_file)

        # Display the uploaded dataset dimensions.
        st.write(
            f"Uploaded dataset shape: {test_data.shape}"
        )

        # Check that all required input columns are available.
        required_columns = FEATURE_COLUMNS + ["y"]

        missing_columns = [
            column
            for column in required_columns
            if column not in test_data.columns
        ]

        # Stop evaluation if required columns are missing.
        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

            st.stop()

        # Extract the original input features.
        X_test = test_data[FEATURE_COLUMNS].copy()

        # Extract and validate the target column.
        y_test = prepare_target(test_data["y"])

        # Apply the previously fitted preprocessing object.
        # Do not fit or retrain the preprocessor on uploaded data.
        X_test_processed = preprocessor.transform(X_test)

        # Display a preview of the uploaded dataset.
        st.subheader("Test Dataset Preview")

        st.dataframe(
            test_data.head(),
            use_container_width=True
        )

        # Display the processed feature dimensions.
        st.write(
            f"Processed test feature shape: {X_test_processed.shape}"
        )

        # Evaluate all five saved models.
        all_results = {}

        all_predictions = {}

        for model_name, model in trained_models.items():

            metrics, predictions = evaluate_model(
                model,
                X_test_processed,
                y_test
            )

            all_results[model_name] = metrics

            all_predictions[model_name] = predictions

        # Convert all model results into a comparison DataFrame.
        comparison_df = pd.DataFrame.from_dict(
            all_results,
            orient="index"
        )

        comparison_df.index.name = "Model"

        # Display the comparison of all five models.
        st.subheader("Model Performance Comparison")

        st.dataframe(
            comparison_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # Allow the user to download the comparison results.
        comparison_csv = comparison_df.to_csv().encode("utf-8")

        st.download_button(
            label="Download Model Comparison CSV",
            data=comparison_csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )

        # Allow the user to select an individual model.
        st.subheader("Individual Model Evaluation")

        selected_model_name = st.selectbox(
            "Select a classification model:",
            list(trained_models.keys())
        )

        # Retrieve the selected model and its predictions.
        selected_model = trained_models[selected_model_name]

        selected_predictions = all_predictions[
            selected_model_name
        ]

        selected_metrics = all_results[
            selected_model_name
        ]

        # Display the selected model's six metrics.
        st.subheader(
            f"{selected_model_name} - Evaluation Metrics"
        )

        metric_columns = st.columns(3)

        metric_names = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]

        for index, metric_name in enumerate(metric_names):

            metric_columns[index % 3].metric(
                metric_name,
                f"{selected_metrics[metric_name]:.4f}"
            )

        # Calculate the confusion matrix.
        cm = confusion_matrix(
            y_test,
            selected_predictions,
            labels=[0, 1]
        )

        # Display the confusion matrix.
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        st.pyplot(fig)

        plt.close(fig)

        # Generate the classification report.
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            selected_predictions,
            target_names=["No (0)", "Yes (1)"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # Display a completion message.
        st.success(
            "Model evaluation completed successfully."
        )

    except Exception as error:

        # Display an understandable error if evaluation fails.
        st.error(
            f"An error occurred while processing the uploaded dataset: "
            f"{error}"
        )

else:

    # Show an instruction before a dataset is uploaded.
    st.info(
        "Please upload test_data.csv to evaluate the saved models."
    )
