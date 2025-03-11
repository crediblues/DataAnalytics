import argparse
from data_preprocessing import fetch_data, preprocess_data
from model import train_model, evaluate_model, save_model

def main(args):
    # Fetch and preprocess the data
    df = fetch_data(args.db_file, args.table_name)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train the model
    model = train_model(X_train, y_train, model_type=args.model_type)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Save the best model
    save_model(model, "best_model.pkl")

    # Print evaluation results
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the machine learning pipeline")
    parser.add_argument('--db_file', type=str, required=True, help="Path to the SQLite database file")
    parser.add_argument('--table_name', type=str, required=True, help="Name of the table in the database")
    parser.add_argument('--model_type', type=str, default='logistic_regression', help="Model type to train")

    args = parser.parse_args()
    main(args)
