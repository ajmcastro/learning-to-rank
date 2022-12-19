import pandas as pd

# Load the training and test data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Preprocess the documents by tokenizing and stemming them
from preprocessing.tokenize import tokenize
from preprocessing.stem import stem

train_docs = train_df["document"].apply(tokenize).apply(stem)
test_docs = test_df["document"].apply(tokenize).apply(stem)

# Convert the relevance labels to numerical values
train_labels = train_df["relevance"].apply(lambda x: 0 if x == "not relevant" else 1)
test_labels = test_df["relevance"].apply(lambda x: 0 if x == "not relevant" else 1)

# Import the RankNet model
from models.ranknet import RankNet

# Create a RankNet model
model = RankNet()

# Train the RankNet model
model.fit(train_docs, train_labels, epochs=10)

# Predict the relevance scores for the test set
predictions = model.predict(test_docs)

# Evaluate the RankNet model using NDCG
from metrics.ndcg import ndcg_at_k

k = 10 # number of documents to consider in the evaluation
ndcg = ndcg_at_k(test_labels, predictions, k)
print(f"NDCG@{k}: {ndcg:.4f}")
