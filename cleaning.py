import pandas as pd
import re

# Load the dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('Reviews.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Function to remove English words from a text
def remove_english_words(text):
    # Regular expression to find English words
    return re.sub(r'\b[A-Za-z]+\b', '', text)

# Assuming the text column is named 'review'
if 'review' in df.columns:
    # Apply the function to the 'review' column
    df['review'] = df['review'].apply(remove_english_words)
    print("English words removed.")
else:
    print("No review column found.")

# Save the modified dataset to a new CSV file
df.to_csv('cleaned_dataset.csv', index=False)

# Display the first few rows of the cleaned dataset
print(df.head())
