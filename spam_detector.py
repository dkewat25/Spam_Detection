import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

DATASET_PATH = 'combined_data.csv' # Make sure this matches your downloaded file name
TEXT_COLUMN = 'text'     # Adjust if your dataset uses a different column name for email content
LABEL_COLUMN = 'label'   # Adjust if your dataset uses a different column name for labels (spam/ham)