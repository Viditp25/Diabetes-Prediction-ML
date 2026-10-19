# Diabetes Prediction Using Machine Learning

## Project Overview

This project implements and compares three machine learning classification algorithms to predict diabetes using the Pima Indians Diabetes Dataset.

The models used are:
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)

The project evaluates model performance using accuracy, precision, recall, F1-score, confusion matrices, and cross-validation.

## Dataset

**Dataset Name:** Pima Indians Diabetes Dataset

**Source:** [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The dataset contains 768 records, eight input features, and one target variable named `Outcome`.

- `Outcome = 0`: Non-diabetic
- `Outcome = 1`: Diabetic

### Input Features

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Files

```text
Diabetes-Prediction-ML/
├── diabetes_prediction_final.py
├── diabetes.csv
└── README.md
```

The Python script contains the data preprocessing, model training, evaluation, comparison, and visualization steps.

## Installation

Install Python 3 and the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Download or clone this repository.
2. Ensure that `diabetes.csv` and `diabetes_prediction_final.py` are in the same directory.
3. Open a terminal in that directory.
4. Run the following command:

```bash
python diabetes_prediction_final.py
```

The program loads the dataset, preprocesses the data, trains the three classification models, evaluates their performance, and generates comparison results and graphs.

## Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted diabetic cases that are actually diabetic.
- **Recall:** Proportion of actual diabetic cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications.
- **Cross-validation:** Evaluates performance consistency across multiple data splits.

## Results and Analysis

The project compares Logistic Regression, Decision Tree, and SVM based on predictive performance, interpretability, computational cost, overfitting, and generalization.

The results are used to identify a suitable model for this dataset. Model selection considers multiple evaluation metrics rather than accuracy alone.

## Conclusion

This project demonstrates the application of supervised machine learning algorithms to diabetes prediction. It compares three classification approaches and analyzes their strengths and limitations using evaluation metrics and visualizations.

**Disclaimer:** This project is for educational purposes only and is not intended for medical diagnosis.

## Author

Machine Learning Digital Assignment

