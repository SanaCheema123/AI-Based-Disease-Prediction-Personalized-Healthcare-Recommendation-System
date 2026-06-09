import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/raw/dataset/Training.csv")

df = df.loc[:, ~df.columns.str.contains("Unnamed")]

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

model = SVC(kernel="linear")

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean())