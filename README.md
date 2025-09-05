# Product Review Sentiment Analyzer

This is a full-stack web application built with Python and Flask that performs sentiment analysis on product reviews. The app features a dynamic dashboard that visualizes the sentiment breakdown and key topics for any product within the dataset.


- **Sentiment Analysis Model:** A machine learning model trained with Scikit-learn to classify reviews as positive or negative with over 91% accuracy.
- **Dynamic Dashboard:** Enter a Product ID (ASIN) to generate an on-demand analysis, including:
    - A pie chart showing the positive vs. negative sentiment ratio.
    - Word clouds for positive and negative reviews to highlight key topics.
- **Model Explainability:** An example review is shown with influential keywords highlighted in green (positive) and red (negative), demonstrating why the model made its decision.
- **Modern UI:** A responsive and visually appealing user interface with a dark theme, animations, and interactive elements.

## How to Run Locally

1.  Clone the repository:
    `git clone https://github.com/YOUR_USERNAME/sentiment-analyzer-flask.git`
2.  Navigate to the project directory:
    `cd sentiment-analyzer-flask`
3.  Create and activate a virtual environment:
    `python -m venv venv`
    `source venv/bin/activate`  4.  Install the required packages:
    `pip install -r requirements.txt`
5.  **Download the Dataset:** This project uses the [Amazon Reviews 2023 dataset from Kaggle](https://www.kaggle.com/datasets/ravirajbabasomane/amazon-reviews-2023). Download `amazon_reviews.csv` and place it in the root of the project folder.
6.  Run the Flask app:
    `python -m flask run`
7.  Open your browser and go to `http://127.0.0.1:5000`.