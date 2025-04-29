import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('twitter_training.csv', encoding='latin1', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'TweetText']

# Drop rows with missing Sentiment or TweetText
df = df.dropna(subset=['Sentiment', 'TweetText'])

# Keep only valid sentiment classes
valid_sentiments = ['Positive', 'Negative', 'Neutral']
df = df[df['Sentiment'].isin(valid_sentiments)]

# Map sentiments to numeric values
sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
df['Sentiment'] = df['Sentiment'].map(sentiment_map)

# Drop rows that might still have NaNs after mapping
df = df.dropna(subset=['Sentiment'])

# Define stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

# Apply cleaning
df['Cleaned_Tweet'] = df['TweetText'].apply(clean_text)

# Features and target
X = df['Cleaned_Tweet']
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])

# Plot
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()