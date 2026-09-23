# Diabetes Prediction using K-Nearest Neighbors

A machine learning project that predicts whether a patient has diabetes based on medical diagnostic measurements.

## Overview

This project applies the K-Nearest Neighbors (KNN) algorithm to the Pima Indians Diabetes dataset. It covers the full machine learning pipeline, from data exploration and preprocessing to model training, evaluation, and visualization.

The main challenge is the presence of hidden missing values. Several medical features such as Glucose, BloodPressure, and BMI contain zero values, which are biologically impossible and must be treated as missing data.

The final model achieved an accuracy of **73.4%** on the test set.

## Dataset

The project uses the **Pima Indians Diabetes Database**, a well-known dataset containing medical records of 768 female patients.

| Property | Value |
|----------|-------|
| Samples | 768 |
| Features | 8 |
| Target | 2 classes (Non-Diabetic, Diabetic) |
| Source | UCI Machine Learning Repository |

The eight input features are Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

## Approach

**1. Exploratory Data Analysis**
Loaded the dataset, inspected its structure, and reviewed descriptive statistics for each feature.

**2. Handling Hidden Missing Values**
Replaced biologically impossible zero values with the median of their respective columns.

**3. Train/Test Split**
Split the dataset into 80% training and 20% testing sets with a fixed random state.

**4. Feature Scaling**
Standardized all features using StandardScaler, since KNN is a distance-based algorithm.

**5. Model Training**
Trained a KNN classifier with the following configuration:

| Parameter | Value |
|-----------|-------|
| n_neighbors | 4 |
| metric | manhattan |
| weights | uniform |

**6. Model Evaluation**
Evaluated the model using accuracy, confusion matrix, and classification report.

## Results

The final model achieved the following performance on the test set:

| Metric | Score |
|--------|-------|
| Accuracy | 73.4% |
| Precision (Diabetic) | 0.67 |
| Recall (Diabetic) | 0.51 |
| F1-Score (Diabetic) | 0.58 |

The model performs better on Non-Diabetic cases than on Diabetic cases. Improving recall for the Diabetic class is an important area for future work, since missing a diabetic patient is more costly than a false positive.

## Visualizations

The notebook includes the following visualizations:

- Confusion Matrix
- ROC Curve
- Effect of K on Error Rate

## Requirements

The project depends on the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

Install them with:

    pip install -r requirements.txt

## How to Run

1. Clone the repository:

       git clone https://github.com/alinedaiy/diabetes-prediction.git

2. Navigate to the project directory:

       cd diabetes-prediction

3. Install the required packages:

       pip install -r requirements.txt

4. Open the notebook:

       jupyter notebook diabetes_knn.ipynb

## Future Work

- Compare KNN with other classifiers such as Logistic Regression, Random Forest, and XGBoost
- Apply cross-validation for more robust hyperparameter tuning
- Explore feature engineering to improve recall on the Diabetic class
- Deploy the model as a web application using Streamlit

## Author

Ali Nedaiy  
GitHub: [alinedaiy](https://github.com/alinedaiy)

## License

This project is licensed under the MIT License.