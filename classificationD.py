import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
try:
    df = pd.read_csv('film.csv')  # Replace with your CSV file path
except FileNotFoundError:
    print("The file was not found. Please check the file path and try again.")
    exit()

# Print the column names to check the correct name for the label column
print("Column names in the dataset:", df.columns)

# Check if the 'review' and 'label' columns exist
if 'review' not in df.columns:
    print("The dataset must contain a 'review' column.")
    exit()

# Assuming the sentiment label column might have a different name
label_column_name = 'label'  # Replace this with the actual column name in your dataset
if label_column_name not in df.columns:
    print(f"The dataset must contain a '{label_column_name}' column.")
    exit()

# Preprocess the data
def preprocess_text(text):
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

df['cleaned_reviews'] = df['review'].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_reviews'])

# Target variable
y = df[label_column_name]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Optional: Use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X.toarray())

df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

# Plot the clusters (if applicable)
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue=label_column_name, data=df, palette='viridis')
plt.title('PCA of Hindi Movie Reviews')
plt.show()

# Print a few example reviews from each sentiment category
for label in df[label_column_name].unique():
    print(f"\n{label.capitalize()} Reviews:")
    print(df[df[label_column_name] == label]['review'].head())
