import pandas as pd
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK stopwords
nltk.download('stopwords')

# ------------------ Load and Preprocess Dataset ------------------

# Load dataset
df = pd.read_csv('twitter_training.csv', encoding='latin1', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'TweetText']

# Drop missing values
df = df.dropna(subset=['Sentiment', 'TweetText'])

# Filter only Positive, Negative, and Neutral sentiments
valid_sentiments = ['Positive', 'Negative', 'Neutral']
df = df[df['Sentiment'].isin(valid_sentiments)]

# Map sentiment labels to numeric
sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
df['Sentiment'] = df['Sentiment'].map(sentiment_map)

# Final check for NaNs
df = df.dropna(subset=['Sentiment'])

# ------------------ Text Cleaning ------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
df['Cleaned_Tweet'] = df['TweetText'].apply(clean_text)

# ------------------ Exploratory Visualizations ------------------

# Add tweet length column
df['Tweet_Length'] = df['TweetText'].apply(len)

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sentiment', y='Tweet_Length', data=df, palette='Set2')
plt.title('Violin Plot of Tweet Length by Sentiment')
plt.xlabel('Sentiment (0=Negative, 1=Neutral, 2=Positive)')
plt.ylabel('Tweet Length')
plt.show()

# Swarm plot (use sample to avoid overplotting)
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Sentiment', y='Tweet_Length', data=df.sample(500), palette='Set1')
plt.title('Swarm Plot of Tweet Length by Sentiment (sample of 500)')
plt.xlabel('Sentiment (0=Negative, 1=Neutral, 2=Positive)')
plt.ylabel('Tweet Length')
plt.show()

# ------------------ Feature Extraction ------------------

X = df['Cleaned_Tweet']
y = df['Sentiment']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------ Model Training & Evaluation ------------------

def evaluate_model(name, model, X_train_vec, y_train, X_test_vec, y_test):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n==== {name} ====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
evaluate_model("Logistic Regression", lr_model, X_train_vec, y_train, X_test_vec, y_test)

# Naive Bayes
nb_model = MultinomialNB()
evaluate_model("Naive Bayes", nb_model, X_train_vec, y_train, X_test_vec, y_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model("Random Forest", rf_model, X_train_vec, y_train, X_test_vec, y_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model("K-Nearest Neighbors", knn_model, X_train_vec, y_train, X_test_vec, y_test)

# ------------------ Confusion Matrix (for final model) ------------------

from sklearn.metrics import ConfusionMatrixDisplay

final_model = lr_model  # or replace with your preferred model
y_pred_final = final_model.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Final Model (Logistic Regression)")
plt.show()
