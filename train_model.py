import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import string

print("Loading dataset...")

# Load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data[["text","label"]]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

print("Cleaning text...")
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# Better vectorizer with more features
vectorizer = TfidfVectorizer(
    max_features=10000, 
    stop_words='english',
    ngram_range=(1, 2),  # Changed from (1, 3) to (1, 2)
    min_df=2,  
    max_df=0.95  
)
print("Converting text to vectors...")
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Better model with more iterations
model = LogisticRegression(
    max_iter=2000,
    C=1.0,  # Regularization
    class_weight='balanced'  # Handle imbalanced data
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy = {accuracy:.4f}")

# Save
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully")