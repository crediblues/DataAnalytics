import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train, model_type="logistic_regression"):
    """Train a machine learning model based on the selected model type."""
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model, filename="best_model.pkl"):
    """Save the trained model to a file."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename="best_model.pkl"):
    """Load a trained model from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)