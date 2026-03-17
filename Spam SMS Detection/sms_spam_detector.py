import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from nltk.corpus import stopwords

nltk.download('stopwords')



df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ["label","message"]

print("Dataset size:", df.shape)



df["label"] = df["label"].map({
    "ham":0,
    "spam":1
})



stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df["clean_message"] = df["message"].apply(clean_text)



X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"],
    df["label"],
    test_size=0.2,
    random_state=42
)



tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)



models = {

    "Naive Bayes":
        MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Support Vector Machine":
        LinearSVC()
}

for name, model in models.items():

    print("\n========================")
    print("Model:", name)

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds))



def predict_sms(text):

    text = clean_text(text)

    vector = tfidf.transform([text])

    prediction = models["Logistic Regression"].predict(vector)

    if prediction[0] == 1:
        return "SPAM MESSAGE"
    else:
        return "LEGITIMATE MESSAGE"



user_sms = input("\nEnter an SMS message: ")

print("Prediction:", predict_sms(user_sms))