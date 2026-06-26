import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# 1. Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Downloading dataset...")
# 2. Using a popular public dataset of Spam (Phishing) vs Ham (Safe) text messages
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])

# 3. Convert labels to numeric: 'ham' (safe) = 1, 'spam' (phishing) = 0
df['label'] = df['label'].map({'ham': 1, 'spam': 0})

X = df['message']
y = df['label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying TF-IDF (Natural Language Processing)...")
# 4. This is the exact NLP technique mentioned in your Chapter 3!
vectorizer = TfidfVectorizer(stop_words='english')
X_train_dtm = vectorizer.fit_transform(X_train)

print("Training Machine Learning Model (Logistic Regression)...")
model = LogisticRegression()
model.fit(X_train_dtm, y_train)

print("Saving Model and Vectorizer...")
# 5. We must save BOTH the model AND the vocabulary translator (vectorizer)
joblib.dump(model, 'models/nlp_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("✅ Training Complete! nlp_model.pkl and vectorizer.pkl saved in the 'models' folder.")