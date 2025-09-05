from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- INITIALIZATION ---
app = Flask(__name__)

# --- LOAD ALL NECESSARY FILES ---
print("Loading model, vectorizer, and dataset...")
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('sentiment_vectorizer.joblib')
full_df = pd.read_csv('amazon_reviews.csv')
full_df.dropna(subset=['asin', 'text'], inplace=True)
print("All files loaded.")

# --- NEW: Create a dictionary of word weights from the model ---
feature_names = vectorizer.get_feature_names_out()
word_weights = dict(zip(feature_names, model.coef_[0]))

# --- TEXT PREPROCESSING FUNCTION (Same as before) ---
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

# --- NEW: Function to highlight influential words ---
def highlight_text(text, weights, threshold=0.5):
    processed_text = preprocess_text(text)
    highlighted_html = ""
    for word in text.split():
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        weight = weights.get(clean_word, 0)
        if weight > threshold:
            highlighted_html += f'<span class="positive-word">{word}</span> '
        elif weight < -threshold:
            highlighted_html += f'<span class="negative-word">{word}</span> '
        else:
            highlighted_html += f'{word} '
    return highlighted_html.strip()

# --- MAIN FLASK ROUTE (Updated) ---
@app.route('/', methods=['GET', 'POST'])
def home():
    analysis_results = None
    if request.method == 'POST':
        product_asin = request.form['product_asin']
        
        product_reviews_df = full_df[full_df['asin'] == product_asin].copy()

        if not product_reviews_df.empty:
            product_reviews_df['text_clean'] = product_reviews_df['text'].apply(preprocess_text)
            text_vectorized = vectorizer.transform(product_reviews_df['text_clean'])
            predictions = model.predict(text_vectorized)
            
            sentiment_counts = pd.Series(predictions).value_counts()
            
            # --- Generate Pie Chart (Same as before) ---
            plt.figure(figsize=(6, 6))
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['#90EE90', '#FFB6C1'])
            plt.title(f'Sentiment for Product: {product_asin}')
            pie_chart_path = os.path.join('static', 'sentiment_pie_chart.png')
            plt.savefig(pie_chart_path)
            plt.close()

            # --- Generate Word Clouds (Same as before) ---
            positive_reviews = ' '.join(product_reviews_df['text_clean'][predictions == 'positive'])
            negative_reviews = ' '.join(product_reviews_df['text_clean'][predictions == 'negative'])
            
            wordcloud_pos_path = None
            if positive_reviews:
                wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
                wordcloud_pos_path = os.path.join('static', 'wordcloud_positive.png')
                wordcloud_pos.to_file(wordcloud_pos_path)
            
            wordcloud_neg_path = None
            if negative_reviews:
                wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_reviews)
                wordcloud_neg_path = os.path.join('static', 'wordcloud_negative.png')
                wordcloud_neg.to_file(wordcloud_neg_path)

            # --- NEW: Get a sample review to highlight ---
            sample_review_text = product_reviews_df['text'].iloc[0]
            highlighted_review = highlight_text(sample_review_text, word_weights)

            analysis_results = {
                'asin': product_asin,
                'total_reviews': len(product_reviews_df),
                'pie_chart': pie_chart_path,
                'wordcloud_pos': wordcloud_pos_path,
                'wordcloud_neg': wordcloud_neg_path,
                'counts': sentiment_counts.to_dict(),
                'highlighted_review': highlighted_review # Pass highlighted review to template
            }
        else:
            analysis_results = {'error': f"No reviews found for Product ID (ASIN): {product_asin}"}

    return render_template('index.html', results=analysis_results)

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)