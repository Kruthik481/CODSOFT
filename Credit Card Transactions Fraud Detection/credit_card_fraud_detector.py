import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)




features = [
    "amt",
    "category",
    "gender",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long"
]

target = "is_fraud"

train = train[features + [target]]
test = test[features + [target]]




encoder = LabelEncoder()

for col in ["category", "gender"]:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])




X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




models = {

    "Logistic Regression":
        LogisticRegression(max_iter=500),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=10),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42
        )
}


for name, model in models.items():

    print("\n==========================")
    print("Model:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds))

    print("ROC-AUC:", roc_auc_score(y_test, preds))




def predict_transaction(data):

    data = scaler.transform([data])

    prediction = models["Random Forest"].predict(data)

    if prediction[0] == 1:
        return "Fraudulent Transaction"
    else:
        return "Legitimate Transaction"


sample_transaction = [
    250.50,  # amount
    2,       # category
    1,       # gender
    50000,   # city_pop
    35.22,   # lat
    -80.84,  # long
    35.21,   # merch_lat
    -80.80   # merch_long
]

print("\nPrediction Example:")
print(predict_transaction(sample_transaction))