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

# Implement the RankNet model
from models.ranknet import RankNet
from models.utils import train, evaluate

model = RankNet()

# Train the RankNet model
train(model, train_docs, train_labels)

# Evaluate the RankNet model on the test set
test_loss, test_acc = evaluate(model, test_docs, test_labels)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Save the trained RankNet model
model.save("ranknet.h5")
