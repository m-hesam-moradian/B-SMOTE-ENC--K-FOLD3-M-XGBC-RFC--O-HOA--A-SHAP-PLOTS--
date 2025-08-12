import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.Utils.getAllMetric import getAllMetric
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, X_test, y_test, params=None):
    # Model selection (same as your current setup)

    default_params = {
        "n_estimators": 50,
        "learning_rate": 0.5,
        "loss": "square",
    }

    final_params = (
        default_params
        if params is None
        else {
            "n_estimators": int(params[0]),
            "learning_rate": float(params[1]),
            "loss": "linear",
        }
    )
    xgb_model = XGBClassifier(
        n_estimators=final_params["n_estimators"],
        learning_rate=final_params["learning_rate"],
        enable_categorical=True,
    )

    rf_model = RandomForestClassifier(
        n_estimators=final_params["n_estimators"],
    )

    # Define stacking classifier
    model = StackingClassifier(
        estimators=[("xgb", xgb_model), ("rf", rf_model)],
        final_estimator=LogisticRegression(max_iter=2000),  # was default 100
        passthrough=True,
    )

    # Fit the model
    model_name = "StackingClassifier(XGB + RF)"

    np.asarray(y_train)

    np.asarray(y_train)
    np.asarray(y_test)
    model.fit(X_train, y_train)
    # Predictions

    y_pred_train = model.predict(X_train)
    midpoint = len(y_test) // 2
    y_pred_test = model.predict(X_test)

    X_value, X_value_test = X_test[:midpoint], X_test[midpoint:]
    y_value, y_value_test = y_test[:midpoint], y_test[midpoint:]

    y_pred_value = model.predict(X_value)
    y_pred_value_test = model.predict(X_value_test)

    metrics_train = pd.DataFrame([getAllMetric(y_train.to_numpy(), y_pred_train)])
    metrics_test = pd.DataFrame([getAllMetric(y_test.to_numpy(), y_pred_test)])
    metrics_value = pd.DataFrame([getAllMetric(y_value.to_numpy(), y_pred_value)])

    metrics_value_test = pd.DataFrame(
        [getAllMetric(y_value_test.to_numpy(), y_pred_value_test)]
    )

    # For concatenated all predictions
    metrics_all = pd.DataFrame(
        [
            getAllMetric(
                np.concatenate([y_train.to_numpy(), y_test.to_numpy()]),
                np.concatenate([y_pred_train, y_pred_test]),
            )
        ]
    )

    # Bundle all metrics
    all_metrics = {
        "all": metrics_all,
        "train": metrics_train,
        "test": metrics_test,
        "value": metrics_value,
        "test_value": metrics_value_test,
    }

    return {
        "model_name": model_name,
        # if there is no params, dont return model
        "model": model,
        # "best_params": pd.DataFrame([final_params]),
        "metrics": all_metrics,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def get_X_y(df, target_col):
    """
    Splits the DataFrame into features (X) and target (y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# def build_stacking_model(xgb_params=None, rf_params=None):
#     if xgb_params is None:
#         xgb = XGBClassifier(eval_metric="logloss", random_state=42)
#     else:
#         xgb = XGBClassifier(
#             learning_rate=xgb_params[0],
#             max_depth=int(xgb_params[1]),
#             n_estimators=int(xgb_params[2]),
#             subsample=xgb_params[3],
#             eval_metric="logloss",
#             random_state=42,
#         )

#     if rf_params is None:
#         rf = RandomForestClassifierRegressor(random_state=42)
#     else:
#         rf = RandomForestClassifierRegressor(
#             n_estimators=int(rf_params[0]),
#             max_depth=int(rf_params[1]),
#             min_samples_split=int(rf_params[2]),
#             min_samples_leaf=int(rf_params[3]),
#             random_state=42,
#         )

#     stacked_model = StackingClassifier(
#         estimators=[("xgb", xgb), ("rf", rf)],
#         final_estimator=LogisticRegression(),
#         cv=5,
#     )

#     return stacked_model


# from sklearn.model_selection import cross_val_score


# def objective_function(
#     params,
#     X_train,
#     y_train,
# ):
#     xgb_params = params[:4]
#     rf_params = params[4:]
#     model = build_stacking_model(xgb_params, rf_params)
#     score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
#     return -score  # Minimize negative accuracy


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# def evaluate_model(
#     model,
#     name,
#     X_train,
#     X_test,
#     y_train,
#     y_test,
# ):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print(f"\n📌 {name} Model Metrics:")
#     print(f"Accuracy  : {acc:.4f}")
#     print(f"Precision : {prec:.4f}")
#     print(f"Recall    : {rec:.4f}")
#     print(f"F1 Score  : {f1:.4f}")
