import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier




df = pd.read_csv("Churn_Modelling.csv")

print("Dataset shape:", df.shape)


df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)




df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)




X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.1
        )
}


for name, model in models.items():

    print("\n============================")
    print("Model:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds))

    print("ROC-AUC:", roc_auc_score(y_test, preds))




rf = RandomForestClassifier(n_estimators=150, random_state=42)

rf.fit(X_train, y_train)

importances = rf.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Important Features:")
print(importance_df.head())




#sample_customer = [[
 #   600,  # CreditScore
  #  0,    # Age
   # 2,    # Tenure
    #50000, # Balance
    #2,    # NumOfProducts
    #1,    # HasCrCard
    #0,    # IsActiveMember
    #70000, # EstimatedSalary
    #1,0,0, # Geography
    #1     # Gender
#]]

#sample_customer = scaler.transform(sample_customer)

#prediction = rf.predict(sample_customer)

#print("\nPrediction Example:")

#if prediction[0] == 1:
 #   print("Customer likely to CHURN")
#else:
 #   print("Customer likely to STAY")