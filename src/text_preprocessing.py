import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (only happens once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    """
    Cleans raw text by removing HTML, punctuation, stopwords, 
    and converting words to their base lemmatized form.
    """
    # 1. Handle missing or non-string data safely
    if not isinstance(text, str):
        return ""
    
    # 2. Remove HTML tags (e.g., <p>, <br>)
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Remove URLs/Hyperlinks
    text = re.sub(r'https?://\s+|www\.\S+', '', text)
    
    # 5. Remove punctuation and special characters (like @, #, $, etc.)
    # This replaces punctuation with an empty space
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 6. Tokenize the text (split sentences into individual words)
    words = word_tokenize(text)
    
    # 7. Get the list of English stopwords
    stop_words = set(stopwords.words('english'))
    
    # 8. Initialize the Lemmatizer (reduces words like 'running' to 'run')
    lemmatizer = WordNetLemmatizer()
    
    # 9. Filter out stopwords and apply lemmatization
    cleaned_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and word.isalnum()
    ]
    
    # 10. Re-join the cleaned words back into a single string
    return " ".join(cleaned_words)


# --- Quick Test Check ---
if __name__ == "__main__":
    sample_text = "<p>The breaking news is that hackers are <b>running</b> away with data! See https://example.com</p>"
    print("Original Text:\n", sample_text)
    print("\nCleaned Text:\n", clean_text(sample_text))