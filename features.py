import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation
import emoji
import pandas as pd

# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Sample DataFrame with text data
data = {'text': ['I love this movie! 😃', 'Hate speech is not acceptable! 😡']}
df = pd.DataFrame(data)

# Tokenize the text, including emojis
def tokenize_with_emojis(text):
    tokens = word_tokenize(text)
    tokens_with_emojis = []
    for token in tokens:
        tokens_with_emojis.extend([emoj for emoj in emoji.demojize(token).split(':') if emoj])
    return tokens_with_emojis

df['tokens'] = df['text'].apply(tokenize_with_emojis)

# Perform sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(tokens):
    # Join the tokens back into a sentence
    text = ' '.join(tokens)
    # Perform sentiment analysis
    sentiment_scores = sid.polarity_scores(mark_negation(text))
    return sentiment_scores

df['sentiment_scores'] = df['tokens'].apply(analyze_sentiment)

# Extract sentiment labels
def extract_sentiment_label(sentiment_scores):
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['sentiment_scores'].apply(extract_sentiment_label)

# Display the DataFrame with sentiment analysis results
print(df[['text', 'tokens', 'sentiment_scores', 'sentiment']])
