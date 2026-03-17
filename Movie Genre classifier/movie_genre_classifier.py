import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


train_file = "train_data.txt"

data = []

with open(train_file, "r", encoding="utf8") as f:
    for line in f:
        parts = line.split(" ::: ")
        if len(parts) >= 4:
            genre = parts[2].strip()
            plot = parts[3].strip()
            data.append([genre, plot])

df = pd.DataFrame(data, columns=["genre", "plot"])

print("Dataset size:", df.shape)



def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["plot"] = df["plot"].apply(clean_text)



X_train, X_test, y_train, y_test = train_test_split(
    df["plot"],
    df["genre"],
    test_size=0.2,
    random_state=42
)



tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)



models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    print("\n====================")
    print(name)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, predictions))



def predict_genre(plot):

    cleaned = clean_text(plot)

    vector = tfidf.transform([cleaned])

    prediction = models["Logistic Regression"].predict(vector)

    return prediction[0]



sample_plot = """
An undercover agent fights a powerful crime
syndicate while trying to save the city
from a deadly terrorist attack.
"""

print("\nPredicted Genre:", predict_genre(sample_plot))