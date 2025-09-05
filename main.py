import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# --- DATA LOADING ---
print("Loading data...")
df = pd.read_csv('amazon_reviews.csv')
df = df[['rating', 'text']]
df.dropna(inplace=True)
df['sentiment'] = np.where(df['rating'] > 3, 'positive', np.where(df['rating'] < 3, 'negative', 'neutral'))
df = df[df['sentiment'] != 'neutral']

# --- TEXT PREPROCESSING ---
print("\nCleaning text data...")
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['text_clean'] = df['text'].apply(preprocess_text)

# --- MODEL BUILDING & TRAINING ---
print("\nBuilding and training the model...")
X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['sentiment'], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=5000 , ngram_range=(1,2))
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# This is the corrected line:
tfidf_test = tfidf_vectorizer.transform(X_test) 

model = LogisticRegression(max_iter=1000)
model.fit(tfidf_train, y_train)
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# --- SAVING THE MODEL AND VECTORIZER ---
print("\nSaving model and vectorizer...")
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf_vectorizer, 'sentiment_vectorizer.joblib')

print("\nFiles saved successfully: sentiment_model.joblib, sentiment_vectorizer.joblib")
print("Process complete!")