# SMS Spam Classifier (TF-IDF + Logistic Regression)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This can be replaced this with real SMS data later
data = {
    "text": [
        "Congratulations! You've won a free ticket",
        "Call me when you reach",
        "Free entry in 2 crore lottery",
        "Are you coming to office today?",
        "You have won $1000 cash prize!"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam"]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
preds = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, preds))

# Example prediction
msg = "Win a brand new car now!"
msg_vec = vectorizer.transform([msg])
print("Prediction:", model.predict(msg_vec)[0])
