import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('film.csv')  # Replace with your CSV file path

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

# Apply KMeans clustering
k = 3  # Number of clusters (you can choose based on your needs)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

# Save each cluster's reviews to separate files
for i in range(k):
    cluster_df = df[df['Cluster'] == i]
    cluster_df.to_csv(f'cluster_{i}_reviews.csv', index=False)

# Optional: Use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X.toarray())

df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
plt.title('KMeans Clustering of Hindi Movie Reviews')
plt.show()

# Print a few example reviews from each cluster
for i in range(k):
    print(f"\nCluster {i} Reviews:")
    print(df[df['Cluster'] == i]['review'].head())
