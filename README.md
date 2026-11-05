# Bank Marketing Subscription Prediction Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on customer-related information and previous marketing campaign details.

The project uses the Bank Marketing dataset and implements five classification algorithms. Their performance is evaluated using multiple classification metrics to identify the most suitable model for the prediction task.

The project also includes an interactive Streamlit application that allows users to upload test data, select a trained classification model, and view its evaluation results without retraining the models.

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Target Variable:** `y`

The dataset contains information about customers contacted during a bank's direct marketing campaigns. The target variable indicates whether a customer subscribed to a term deposit.

- `yes` indicates that the customer subscribed.
- `no` indicates that the customer did not subscribe.

### Dataset characteristics

| Attribute | Description |
|---|---|
| Dataset name | Bank Marketing |
| Total instances | 45,211 |
| Total original columns | 17 |
| Input features used | 15 |
| Target variable | `y` |
| Task type | Binary Classification |
| Positive class | Yes (1) |
| Negative class | No (0) |

The `duration` feature was excluded from model training to avoid relying on call duration, which is only known after a marketing call has taken place. The remaining 15 input features were used for classification.

### Input features

1. age
2. job
3. marital
4. education
5. default
6. balance
7. housing
8. loan
9. contact
10. day
11. month
12. campaign
13. pdays
14. previous
15. poutcome

The target column is `y`.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/Souradip-Paul/ML_Assignment_2.git

## 4. Machine Learning Models Used

Five classification algorithms were implemented and evaluated using the same Bank Marketing dataset.

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbour (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

A fitted preprocessing object was used to apply the required data transformations consistently. The trained models and preprocessing object were saved as Joblib files for subsequent use in the Streamlit application.

## 5. Model Performance Comparison

The five trained models were evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC:** Measures the model's ability to distinguish between the two classes using prediction scores.
- **Precision:** Measures the proportion of predicted positive cases that are actually positive.
- **Recall:** Measures the proportion of actual positive cases correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications while considering all four confusion matrix outcomes.

### Comparative results

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 89.33% | 77.17% | 66.32% | 17.86% | 28.15% | 0.3065 |
| Decision Tree | 83.09% | 61.32% | 29.82% | 32.89% | 31.28% | 0.2170 |
| KNN | 88.74% | 70.27% | 55.29% | 19.75% | 29.11% | 0.2833 |
| Naive Bayes | 84.52% | 75.14% | 37.08% | 46.41% | 41.23% | 0.3271 |
| Support Vector Machine | 89.32% | 76.82% | 65.13% | 18.71% | 29.07% | 0.3101 |

## 6. Observations on Model Performance

### 6.1 Logistic Regression

- Achieved the highest accuracy of 89.33%.
- Recorded the highest AUC of 77.17% and precision of 66.32%.
- However, its recall was relatively low at 17.86%.
- The model demonstrates good overall accuracy and precision but identifies a relatively small proportion of actual positive cases.

### 6.2 Decision Tree

- Achieved an accuracy of 83.09% and an AUC of 61.32%.
- Its recall was 32.89%, higher than Logistic Regression, KNN, and SVM.
- Its F1 score was 31.28%.
- It recorded the lowest MCC of 0.2170 among the five models.

### 6.3 K-Nearest Neighbour (KNN)

- Achieved an accuracy of 88.74%.
- Recorded an AUC of 70.27% and precision of 55.29%.
- Its recall was relatively low at 19.75%.
- Its F1 score of 29.11% indicates that the balance between precision and recall could be improved.

### 6.4 Naive Bayes

- Achieved the highest recall of 46.41%.
- Recorded the highest F1 score of 41.23%.
- Achieved the highest MCC of 0.3271.
- Its AUC was 75.14%, indicating competitive class discrimination.
- Although its accuracy was lower than Logistic Regression, KNN, and SVM, it demonstrated the strongest balance between precision and recall.

### 6.5 Support Vector Machine (SVM)

- Achieved an accuracy of 89.32%, close to the highest-performing model.
- Recorded an AUC of 76.82% and precision of 65.13%.
- Its recall was 18.71%, indicating that it missed a considerable proportion of actual positive cases.
- Its F1 score was 29.07%.

## 7. Overall Best Performing Model

Based on the comparative evaluation, **Naive Bayes is selected as the overall best-performing model** among the five classifiers.

The selection is based on the following results:

- Highest Recall: 46.41%
- Highest F1 Score: 41.23%
- Highest MCC: 0.3271
- Competitive AUC: 75.14%

Although Logistic Regression achieved the highest accuracy, accuracy alone does not provide a complete assessment of model performance, particularly when the objective is to identify customers who subscribe to a term deposit.

Naive Bayes provides a comparatively better balance between identifying positive cases and overall classification quality based on the reported metrics.

## 8. Streamlit Application

An interactive web application was developed using Streamlit to evaluate the five trained classification models on an uploaded test dataset.

### Application features

1. Upload a test dataset in CSV format.
2. Validate the presence of all 15 required input features and the target column `y`.
3. Load the five previously trained classification models.
4. Load the fitted preprocessing object.
5. Apply the existing preprocessing transformations to the uploaded test data.
6. Evaluate all five models using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
7. Display a comparative performance table for all five models.
8. Select an individual classification model through a dropdown.
9. Display the selected model's six evaluation metrics.
10. Display the confusion matrix.
11. Display the classification report.
12. Download the model comparison results as a CSV file.

**Important:** The application uses saved models and the fitted preprocessing object. It does not retrain the models using the uploaded test dataset.

### Live Streamlit Application

**Streamlit App URL:** https://mlassignment2-hskvtevrpqxnsj4qdnr9wx.streamlit.app/

## 9. Project Structure

The GitHub repository contains the following project files:

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── preprocessor.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

### Description of important files

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit application source code |
| `requirements.txt` | Python dependencies required to run the application |
| `README.md` | Project documentation |
| `test_data.csv` | Test dataset used for model evaluation |
| `model_comparison.csv` | Comparative performance results |
| `model/` | Directory containing saved models and preprocessing artifact |
| `preprocessor.joblib` | Saved fitted preprocessing object |
| Model `.joblib` files | Saved trained classification models |

## 10. Technologies and Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 11. Conclusion

This project demonstrates an end-to-end machine learning classification workflow using the Bank Marketing dataset. Five classification algorithms were trained and evaluated using six performance metrics.

The comparative results show that Logistic Regression achieved the highest accuracy, while Naive Bayes achieved the highest recall, F1 score, and MCC. Considering the objective of identifying potential term deposit subscribers, Naive Bayes was selected as the overall best-performing model based on the available evaluation results.

Finally, a Streamlit application was developed to make the trained models accessible through an interactive interface. Users can upload test data, compare model performance, select individual classifiers, and view evaluation metrics, confusion matrices, and classification reports without retraining the models.
