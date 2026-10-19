import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Load the dataset. Keep diabetes.csv in the same folder as this script.
df = pd.read_csv("diabetes.csv")
print("Dataset shape:", df.shape)
print("Explicit missing values per column:")
print(df.isna().sum())
print("Class distribution:")
print(df["Outcome"].value_counts().sort_index())

feature_cols = [c for c in df.columns if c != "Outcome"]
X = df[feature_cols].copy()
y = df["Outcome"].astype(int)

# In this dataset, zeros in these columns are treated as invalid measurements.
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("Zero-coded measurements:")
print(X[zero_cols].eq(0).sum())
for col in zero_cols:
    X[col] = X[col].replace(0, np.nan)

# 2. Stratified 80:20 hold-out split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def make_pipeline(estimator, scale=False):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_selection", SelectKBest(score_func=f_classif, k=6))
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)

models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=2000), scale=True
    ),
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(max_depth=5, random_state=42), scale=False
    ),
    "SVM": make_pipeline(SVC(kernel="rbf", C=1.0), scale=True)
}

# 3. Five-fold CV on training data only; do not use the hold-out test set to select models.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "recall": "recall", "f1": "f1"}
results = {}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    pred = model.predict(X_test)
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring,
        return_train_score=True
    )

    results[name] = {
        "Training accuracy": accuracy_score(y_train, model.predict(X_train)),
        "Test accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "Confusion matrix": confusion_matrix(y_test, pred),
        "Fit time (seconds)": fit_seconds,
        "CV accuracy mean": cv_results["test_accuracy"].mean(),
        "CV accuracy SD": cv_results["test_accuracy"].std(ddof=1),
        "CV recall mean": cv_results["test_recall"].mean(),
        "CV recall SD": cv_results["test_recall"].std(ddof=1),
        "CV F1 mean": cv_results["test_f1"].mean(),
        "CV F1 SD": cv_results["test_f1"].std(ddof=1)
    }

    print("\n", name)
    for key in ["Training accuracy", "Test accuracy", "Precision", "Recall", "F1-score", "Fit time (seconds)"]:
        print(f"{key}: {results[name][key]:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(results[name]["Confusion matrix"])
    print(classification_report(y_test, pred, target_names=["Non-diabetic", "Diabetic"], zero_division=0))
    print("5-fold CV on training data:")
    print(f"  Accuracy: {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std(ddof=1):.4f}")
    print(f"  Recall:   {cv_results['test_recall'].mean():.4f} +/- {cv_results['test_recall'].std(ddof=1):.4f}")
    print(f"  F1-score: {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std(ddof=1):.4f}")

# 4. Selected features and ANOVA F-scores, fitted on the training set.
selector = models["Logistic Regression"].named_steps["feature_selection"]
selected = np.array(feature_cols)[selector.get_support()]
scores = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False)
print("\nSelected features:", list(selected))
print("ANOVA F-scores for selected features:")
print(scores.loc[selected].sort_values(ascending=False))

# 5. Logistic Regression coefficients (coefficients are for standardized features).
lr = models["Logistic Regression"]
coef = pd.Series(lr.named_steps["classifier"].coef_[0], index=selected)
print("\nLogistic Regression coefficients:")
print(coef.sort_values(key=np.abs, ascending=False))

# 6. Decision Tree feature importances.
tree = models["Decision Tree"]
importance = pd.Series(
    tree.named_steps["classifier"].feature_importances_, index=selected
).sort_values(ascending=False)
print("\nDecision Tree feature importances:")
print(importance)

# 7. Graph: test-set metric comparison.
metric_names = ["Test accuracy", "Precision", "Recall", "F1-score"]
keys = ["Test accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(models))
width = 0.19
plt.figure(figsize=(8.5, 4.8))
for i, metric in enumerate(metric_names):
    plt.bar(x + (i - 1.5) * width, [results[m][metric] for m in models],
            width, label=metric)
plt.xticks(x, list(models), rotation=10, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Test-set performance comparison")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=180)
plt.close()

# 8. Graph: actual confusion matrices.
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, name in zip(axes, models):
    cm = results[name]["Confusion matrix"]
    ax.imshow(cm)
    ax.set_title(name, fontsize=9)
    ax.set_xticks([0, 1], ["Non-diabetic", "Diabetic"], rotation=20, ha="right", fontsize=7)
    ax.set_yticks([0, 1], ["Non-diabetic", "Diabetic"], fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.suptitle("Confusion matrices: rows = actual, columns = predicted")
fig.tight_layout()
fig.savefig("confusion_matrices.png", dpi=180)
plt.close(fig)

# 9. Graph: cross-validation recall with standard deviation.
plt.figure(figsize=(7.5, 4.2))
means = [results[m]["CV recall mean"] for m in models]
sds = [results[m]["CV recall SD"] for m in models]
plt.bar(list(models), means, yerr=sds, capsize=5)
plt.ylim(0, 1)
plt.ylabel("Recall (mean ± standard deviation)")
plt.title("Five-fold stratified cross-validation: recall")
plt.xticks(rotation=8, ha="right")
plt.tight_layout()
plt.savefig("cross_validation_recall.png", dpi=180)
plt.close()

# 10. Graph: Logistic Regression coefficients and Decision Tree.
plt.figure(figsize=(7.5, 4.2))
coef.sort_values().plot(kind="barh")
plt.axvline(0, linewidth=0.8)
plt.xlabel("Coefficient (standardized features)")
plt.title("Logistic Regression coefficients")
plt.tight_layout()
plt.savefig("logistic_coefficients.png", dpi=180)
plt.close()

plt.figure(figsize=(13, 7))
plot_tree(tree.named_steps["classifier"], feature_names=list(selected),
          class_names=["Non-diabetic", "Diabetic"], filled=True, rounded=True,
          fontsize=8, max_depth=3)
plt.title("Decision Tree (display truncated at depth 3)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=160)
plt.close()

# 11. Save the numerical comparison table.
summary = pd.DataFrame({
    name: {k: v for k, v in res.items() if k != "Confusion matrix"}
    for name, res in results.items()
}).T
summary.to_csv("model_metrics.csv")
print("\nSaved: model_comparison.png, confusion_matrices.png, "
      "cross_validation_recall.png, logistic_coefficients.png, "
      "decision_tree.png, model_metrics.csv")
