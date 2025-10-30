import os
import re
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import nltk
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)


# --- Ensure NLTK stopwords are available ---
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

# --- Function to clean text ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
    text = re.sub(r'\d+', '', text)          # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# --- Function to get sentiment ---
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# --- Main program ---
def main():
    input_file = 'course_feedback.csv'
    output_file = 'analyzed_feedback.csv'

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found in current directory.")
        return

    # Load data
    print("📂 Loading feedback data...")
    df = pd.read_csv(input_file)

    if 'feedback' not in df.columns:
        print("❌ Error: CSV must contain a 'feedback' column.")
        return

    # Clean text
    print("🧹 Cleaning feedback text...")
    df['cleaned_feedback'] = df['feedback'].apply(clean_text)

    # Sentiment analysis
    print("🧠 Performing sentiment analysis...")
    df['sentiment'] = df['cleaned_feedback'].apply(get_sentiment)

    # Keyword extraction
    print("🔍 Extracting keywords using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=20)
    X = vectorizer.fit_transform(df['cleaned_feedback'])
    keywords = vectorizer.get_feature_names_out()
    print("\nTop Keywords:", ", ".join(keywords))

    # Display top keywords with TF-IDF scores
    import numpy as np
    tfidf_scores = np.mean(X.toarray(), axis=0)
    keyword_df = pd.DataFrame({'Keyword': keywords, 'Score': tfidf_scores})
    keyword_df = keyword_df.sort_values(by='Score', ascending=False)
    print("\nTop 10 Keywords by TF-IDF Score:")
    print(keyword_df.head(10).to_string(index=False))

    # Sentiment chart
    print("📊 Generating sentiment chart...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='sentiment', data=df, palette='Set2')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Feedbacks')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300)
    plt.show()

    # Save analyzed feedback
    df.to_csv(output_file, index=False)
    print(f"\n✅ Analysis complete! Results saved to '{output_file}'")
    print("📈 Sentiment chart saved as 'sentiment_distribution.png'")

# --- Run the script ---
if __name__ == "__main__":
    main()
