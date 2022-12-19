import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split the dataset into a training set and a test set."""
    # Generate a random permutation of the indices of the examples
    permutation = np.random.permutation(X.shape[0])
    # Split the indices into a training set and a test set
    split = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[permutation[:split]], X[permutation[split:]]
    y_train, y_test = y[permutation[:split]], y[permutation[split:]]
    return X_train, X_test, y_train, y_test

def evaluate(model, X, y):
    """Evaluate a model on a dataset and return the accuracy."""
    # Make predictions on the test set
    y_pred = model.predict(X)
    # Compute the accuracy
    accuracy = np.mean(y_pred == y)
    return accuracy

def train_and_evaluate(model, X, y):
    """Train a model on a dataset and evaluate it."""
    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Fit the model on the training set
    model.fit(X_train, y_train)
    # Evaluate the model on the test set
    accuracy = evaluate(model, X_test, y_test)
    return accuracy
