import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ===============================
# 1. Paths
# ===============================
TRAIN_PATH = "data/raw/dataset/Training.csv"
TEST_PATH = "data/raw/dataset/Testing.csv"

MODEL_DIR = "models"
PLOT_DIR = "results/plots"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ===============================
# 2. Load Dataset
# ===============================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Remove unwanted column
train_df = train_df.loc[:, ~train_df.columns.str.contains("Unnamed")]

print("Training Shape:", train_df.shape)
print("Testing Shape:", test_df.shape)

# ===============================
# 3. Split Features and Target
# ===============================
X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]

X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

# Encode disease labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# ===============================
# 4. Models
# ===============================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None
    ),
    "SVM": SVC(kernel="linear", probability=True, random_state=42)
}

results = []


# ===============================
# 5. Train and Evaluate
# ===============================
best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average="weighted")
    recall = recall_score(y_test_encoded, y_pred, average="weighted")
    f1 = f1_score(y_test_encoded, y_pred, average="weighted")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name

    # Save classification report
    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_
    )

    with open(f"{REPORT_DIR}/{name}_classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test_encoded, y_pred)

    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_confusion_matrix.png", dpi=300)
    plt.close()


# ===============================
# 6. Save Results
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison_results.csv", index=False)

print("\nModel Comparison:")
print(results_df)

# Bar plot comparison
plt.figure(figsize=(10, 6))
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/model_performance_comparison.png", dpi=300)
plt.close()


# ===============================
# 7. Save Best Model
# ===============================
joblib.dump(best_model, f"{MODEL_DIR}/best_disease_model.pkl")
joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
joblib.dump(list(X_train.columns), f"{MODEL_DIR}/symptom_columns.pkl")

print(f"\nBest Model: {best_model_name}")
print(f"Best F1 Score: {best_score:.4f}")
print("Model saved successfully.")