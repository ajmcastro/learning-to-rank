import nltk

# Download the list of stopwords in English
nltk.download('stopwords')

# Load the list of stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Define a function to remove stopwords from a list of words
def remove_stopwords(words):
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    return filtered_words

# Example usage
words = ["this", "is", "a", "sentence", "with", "some", "stopwords"]
filtered_words = remove_stopwords(words)
print(filtered_words)

# Output: ['sentence', 'stopwords']
