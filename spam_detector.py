import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import joblib  # For saving the model

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Clean line breaks
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Preprocessing - clean, remove punctuation, stopwords, and apply stemming
corpus = []
for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

# Convert text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df['label_num']

# Train-test split with reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42)
clf.fit(X_train, Y_train)

# Evaluate the model
print("Model Accuracy:", clf.score(X_test, Y_test))
y_pred = clf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(Y_test, y_pred, target_names=['Ham', 'Spam']))

# Save the model and vectorizer
joblib.dump(clf, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Predict a sample email from the dataset
email_to_classify = df['text'].iloc[10]
print("\nEmail to classify:\n", email_to_classify)

# Preprocess the email
email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
email_text = ' '.join(email_text)

# Vectorize the email
x_email = vectorizer.transform([email_text])

# Predict and display the result
label_map = {0: 'Ham', 1: 'Spam'}
prediction = clf.predict(x_email)[0]
print("Prediction:", label_map[prediction])
