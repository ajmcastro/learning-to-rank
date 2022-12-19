import nltk
from nltk.stem import PorterStemmer

# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Define a function to stem a list of words
def stem_words(words):
    # Stem the words
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return stemmed_words

# Example usage
words = ["running", "runs", "run"]
stemmed_words = stem_words(words)
print(stemmed_words)

# Output: ['run', 'run', 'run']
