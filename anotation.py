import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from googletrans import Translator

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv('film (1).csv')

# Display the first few rows of the dataset to check the column names
print(df.head())

# Display the column names in the dataset
print(df.columns)

# Ensure there are no missing values in the relevant columns
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Check if 'review' column exists and preprocess it
if 'review' in df.columns:
    df['cleaned_review'] = df['review'].apply(preprocess_text)
else:
    print("The 'review' column is not found in the dataset. Please check the column names.")

# Translate Hindi reviews to English
translator = Translator()

def translate_to_english(text):
    try:
        translated = translator.translate(text, src='hi', dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

df['translated_review'] = df['cleaned_review'].apply(translate_to_english)

# Define a function to get sentiment using TextBlob
def get_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment analysis to the translated reviews if the column exists
if 'translated_review' in df.columns:
    df['label'] = df['translated_review'].apply(get_sentiment)
    # Save the annotated dataset
    df.to_csv('annotated_reviews.csv', index=False)
    # Display the first few rows of the annotated dataset
    print(df.head())
else:
    print("The 'translated_review' column was not created. Please check the preprocessing steps.")
