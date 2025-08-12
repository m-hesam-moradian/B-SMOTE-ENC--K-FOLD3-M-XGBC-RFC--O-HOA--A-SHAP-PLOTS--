from src.model.train_model import train_model


def objective_stacking(params, X_train, y_train, X_test, y_test):
    """
    Objective function for optimizing StackingClassifier(XGB + RF) using your train_model.

    Parameters:
        params[0] -> n_estimators (int)
        params[1] -> learning_rate (float)
    """

    # Train the model using your existing function
    result = train_model(X_train, y_train, X_test, y_test, params=params)

    # Choose a metric to optimize — e.g., 'accuracy', 'f1', 'rmse', etc.
    # Let's assume you want to minimize RMSE from test set:
    rmse = result["metrics"]["test"]["RMSE"].values[
        0
    ]  # make sure 'rmse' exists in your getAllMetric

    return rmse  # This scalar value will be minimized by HOA optimizer
