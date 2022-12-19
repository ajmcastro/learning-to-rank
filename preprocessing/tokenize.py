import nltk
from nltk.stem import PorterStemmer

# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Define a function to stem and tokenize a document
def stem_and_tokenize(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens

# Example usage
text = "This is a sample document that we want to tokenize and stem."
tokens = stem_and_tokenize(text)
print(tokens)

# Output: ['thi', 'is', 'a', 'sampl', 'document', 'that', 'we', 'want', 'to', 'token', 'and', 'stem', '.']
