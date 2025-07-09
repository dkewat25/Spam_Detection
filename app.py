import streamlit as st
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if not already present
nltk.download('stopwords')

# Load model and vectorizer
clf = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

# Streamlit UI
st.title("📧 Spam-Ham Classifier")
st.markdown("Enter an email message below and click **Classify** to know whether it's **Spam** or **Ham**.")

# Text input
email_input = st.text_area("✉️ Email Text", height=200)

# Button
if st.button("🔍 Classify"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        processed = preprocess_text(email_input)
        vectorized = vectorizer.transform([processed])
        prediction = clf.predict(vectorized)[0]

        # Display result
        label = {0: 'Ham', 1: 'Spam'}[prediction]
        if prediction == 1:
            st.error("🚫 This is classified as SPAM.")
        else:
            st.success("✅ This is classified as HAM (Not Spam).")
